# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import asyncio
import random
import time
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request

from lmdeploy.serve.core.chat_runner import ChatRunner, ChatRunnerOptions
from lmdeploy.serve.core.exceptions import RequestError
from lmdeploy.serve.openai.chat_completions.logits_processors import logit_bias_logits_processor
from lmdeploy.serve.openai.chat_completions.logprobs import _create_chat_completion_logprobs
from lmdeploy.serve.openai.errors import create_error_response, create_request_error_response
from lmdeploy.serve.openai.extra_protocol import (BatchChatCompletionRequest, BatchChatCompletionResponse,
                                                  BatchChatCompletionResponseChoice)
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, ChatMessage, UsageInfo
from lmdeploy.serve.openai.utils import get_model_list
from lmdeploy.serve.utils.server_utils import validate_json_request
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def register(router: APIRouter, server_context) -> None:

    @router.post('/v1/chat/batch_completions', dependencies=[Depends(validate_json_request)])
    async def batch_chat_completions_v1(request: BatchChatCompletionRequest, raw_request: Request = None):
        """Batch variant of ``/v1/chat/completions``.

        ``messages`` carries a batch of chat histories, e.g.
        ``[[{"role": "user", "content": "hi"}], [{"role": "user", "content": "bye"}]]``.
        Each history is inferred independently and concurrently (non-streaming),
        sharing the request-level sampling parameters. Returns one choice per
        input history plus an aggregated usage.
        """
        model_name = request.model
        if model_name not in get_model_list(server_context):
            return create_error_response(HTTPStatus.NOT_FOUND, f'The model {model_name!r} does not exist.')

        adapter_name = None
        if model_name != server_context.async_engine.model_name:
            adapter_name = model_name  # got an adapter name
        created_time = int(time.time())

        # normalize `messages` into a batch (list of chat histories)
        batch_messages = request.messages or []
        if batch_messages and not isinstance(batch_messages[0], list):
            batch_messages = [batch_messages]

        tokenizer = server_context.async_engine.tokenizer.model.model
        # sampling params shared by every item; drop batch-only / per-item fields
        base_fields = request.model_dump(exclude={'messages', 'session_id', 'stream'})

        async def _infer_one(index: int, messages) -> BatchChatCompletionResponseChoice:
            cc_request = ChatCompletionRequest(
                **base_fields,
                messages=messages,
                stream=False,
                session_id=random.randint(1, 2**31 - 1),
            )
            logits_processors = None
            if cc_request.logit_bias is not None:
                logits_processors = [logit_bias_logits_processor(cc_request.logit_bias, tokenizer)]
            chat_runner = await ChatRunner.prepare(
                server_context,
                cc_request,
                ChatRunnerOptions(
                    do_preprocess=True,
                    adapter_name=adapter_name,
                    gen_config_kwargs=dict(
                        logits_processors=logits_processors,
                        random_seed=cc_request.seed,
                    ),
                ),
            )
            try:
                res = await chat_runner.collect()
            finally:
                await chat_runner.close()

            logprobs = None
            if cc_request.logprobs and len(res.logprobs):
                logprobs = _create_chat_completion_logprobs(tokenizer, res.token_ids, res.logprobs)
            choice = BatchChatCompletionResponseChoice(
                index=index,
                message=ChatMessage(role='assistant', content=res.text),
                logprobs=logprobs,
                finish_reason=res.finish_reason,
            )
            choice.usage = UsageInfo.build(
                prompt_tokens=res.input_token_len,
                completion_tokens=res.generate_token_len,
                cached_tokens=res.cached_tokens,
                reasoning_tokens=res.reasoning_tokens,
            )
            return choice

        logger.info(f'batch chat completion request: {len(batch_messages)} items')

        try:
            choices = await asyncio.gather(*[_infer_one(i, msgs) for i, msgs in enumerate(batch_messages)])
        except RequestError as error:
            return create_request_error_response(error)

        sum_prompt_tokens = sum(c.usage.prompt_tokens for c in choices)
        sum_completion_tokens = sum(c.usage.completion_tokens for c in choices)
        response = BatchChatCompletionResponse(
            id=str(request.session_id),
            created=created_time,
            model=model_name,
            choices=list(choices),
            usage=UsageInfo(
                prompt_tokens=sum_prompt_tokens,
                completion_tokens=sum_completion_tokens,
                total_tokens=sum_prompt_tokens + sum_completion_tokens,
            ),
        )
        return response
