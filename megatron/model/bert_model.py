# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT model."""

import torch

from megatron import get_args
from megatron import mpu
from megatron.model.language_model import parallel_lm_logits
from megatron.model.language_model import get_language_model
from megatron.model import import_layernorm
from megatron.model.utils import openai_gelu, erf_gelu
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal
from .module import MegatronModule

def bert_attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores

def bert_extended_attention_mask(attention_mask):
    # We create a 3D attention mask from a 2D tensor mask.
    # [b, 1, s]
    attention_mask_b1s = attention_mask.unsqueeze(1)
    # [b, s, 1]
    attention_mask_bs1 = attention_mask.unsqueeze(2)
    # [b, s, s]
    attention_mask_bss = attention_mask_b1s * attention_mask_bs1
    # [b, 1, s, s]
    extended_attention_mask = attention_mask_bss.unsqueeze(1)

    # Convert attention mask to binary:
    extended_attention_mask = (extended_attention_mask < 0.5)

    return extended_attention_mask

def bert_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids


class BertLMHead(MegatronModule):
    """Masked LM head for Bert

    Arguments:
        mpu_vocab_size: model parallel size of vocabulary.
        hidden_size: hidden size
        init_method: init method for weight initialization
        layernorm_epsilon: tolerance for layer norm divisions
        parallel_output: whether output logits being distributed or not.
    """

    def __init__(self, mpu_vocab_size, hidden_size, init_method,
                 layernorm_epsilon, parallel_output):

        super(BertLMHead, self).__init__()

        args = get_args()

        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        self.bias.tensor_model_parallel = True
        self.bias.partition_dim = 0
        self.bias.stride = 1
        self.parallel_output = parallel_output

        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)
        LayerNorm = import_layernorm(args.fp32_residual_connection)
        self.layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.gelu = torch.nn.functional.gelu
        if args.openai_gelu:
            self.gelu = openai_gelu
        elif args.onnx_safe:
            self.gelu = erf_gelu

    def forward(self, hidden_states, word_embeddings_weight):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        output = parallel_lm_logits(hidden_states,
                                    word_embeddings_weight,
                                    self.parallel_output,
                                    bias=self.bias)
        return output


def post_language_model_processing(lm_output, pooled_output,
                                   lm_head, binary_head,
                                   lm_labels,
                                   logit_weights,
                                   fp16_lm_cross_entropy):
    # Output.
    lm_logits = lm_head(
        lm_output, logit_weights)

    binary_logits = None
    if binary_head is not None:
        binary_logits = binary_head(pooled_output)

    if lm_labels is None:
        return lm_logits, binary_logits
    else:
        if fp16_lm_cross_entropy:
            assert lm_logits.dtype == torch.half
            lm_loss = mpu.vocab_parallel_cross_entropy(lm_logits, lm_labels)
        else:
            lm_loss = mpu.vocab_parallel_cross_entropy(lm_logits.float(),
                                                       lm_labels)
        return lm_loss, binary_logits


class BertModelBase(MegatronModule):
    """Bert Language model."""

    def __init__(self, num_tokentypes=2, add_binary_head=True,
                 parallel_output=True):
        super(BertModelBase, self).__init__()
        args = get_args()

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.add_binary_head = add_binary_head
        self.parallel_output = parallel_output

        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

        self.language_model, self._language_model_key = get_language_model(
            attention_mask_func=bert_attention_mask_func,
            num_tokentypes=num_tokentypes,
            add_pooler=self.add_binary_head,
            init_method=init_method,
            scaled_init_method=scaled_init_method)

        self.initialize_word_embeddings(init_method_normal)
        if mpu.is_pipeline_last_stage():
            self.lm_head = BertLMHead(
                self.word_embeddings_weight().size(0),
                args.hidden_size, init_method, args.layernorm_epsilon, parallel_output)
            self._lm_head_key = 'lm_head'
            self.binary_head = None
            if self.add_binary_head:
                self.binary_head = get_linear_layer(args.hidden_size, 2,
                                                    init_method)
                self._binary_head_key = 'binary_head'

    def forward(self, bert_model_input, attention_mask,
                tokentype_ids=None, lm_labels=None):

        extended_attention_mask = bert_extended_attention_mask(attention_mask)

        kwargs = {}
        if mpu.is_pipeline_first_stage():
            input_ids = bert_model_input
            position_ids = bert_position_ids(input_ids)
            args = [input_ids, position_ids, extended_attention_mask]
            kwargs['tokentype_ids'] = tokentype_ids
        else:
            args = [bert_model_input, extended_attention_mask]
        lm_output = self.language_model(*args, **kwargs)
        if mpu.is_pipeline_last_stage() and self.add_binary_head:
            lm_output, pooled_output = lm_output
        else:
            pooled_output = None

        return lm_output


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars)
        if mpu.is_pipeline_last_stage():
            state_dict_[self._lm_head_key] \
                = self.lm_head.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        if mpu.is_pipeline_last_stage() and self.add_binary_head:
            state_dict_[self._binary_head_key] \
                = self.binary_head.state_dict(destination, prefix, keep_vars)
        # Save word_embeddings.
        if mpu.is_pipeline_last_stage() and not mpu.is_pipeline_first_stage():
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if mpu.is_pipeline_last_stage():
            self.lm_head.load_state_dict(
                state_dict[self._lm_head_key], strict=strict)
        if mpu.is_pipeline_last_stage() and self.add_binary_head:
            self.binary_head.load_state_dict(
                state_dict[self._binary_head_key], strict=strict)
        # Load word_embeddings.
        if mpu.is_pipeline_last_stage() and not mpu.is_pipeline_first_stage():
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)


class BertModel(BertModelBase):

    def __init__(self, num_tokentypes=2, add_binary_head=True,
                 parallel_output=True):
        super(BertModel, self).__init__(
            num_tokentypes=num_tokentypes,
            add_binary_head=add_binary_head,
            parallel_output=parallel_output)

    def forward(self, input_ids, attention_mask,
                tokentype_ids=None, lm_labels=None):
        return super(BertModel, self).forward(
            input_ids,
            attention_mask,
            tokentype_ids=tokentype_ids,
            lm_labels=lm_labels)


class BertModelFirstStage(BertModelBase):

    def __init__(self, num_tokentypes=2):
        super(BertModelFirstStage, self).__init__(
            num_tokentypes=num_tokentypes)

    def forward(self, input_ids, attention_mask,
                tokentype_ids=None):
        return super(BertModelFirstStage, self).forward(
            input_ids,
            attention_mask,
            tokentype_ids=tokentype_ids)


class BertModelIntermediateStage(BertModelBase):

    def __init__(self, num_tokentypes=2):
        super(BertModelIntermediateStage, self).__init__(
            num_tokentypes=num_tokentypes)

    def forward(self, hidden_state, attention_mask):
        return super(BertModelIntermediateStage, self).forward(
            hidden_state,
            attention_mask)


class BertModelLastStage(BertModelBase):

    def __init__(self, num_tokentypes=2, add_binary_head=True,
                 parallel_output=True):
        super(BertModelLastStage, self).__init__(
            num_tokentypes=num_tokentypes,
            add_binary_head=add_binary_head,
            parallel_output=parallel_output)

    def forward(self, hidden_state, attention_mask,
                lm_labels=None):
        return super(BertModelLastStage, self).forward(
            hidden_state,
            attention_mask,
            lm_labels=lm_labels)
