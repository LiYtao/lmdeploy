# Copyright (c) OpenMMLab. All rights reserved.
"""yijian platform integration: default model-name injection and optional
BCE-based authentication / subscription / resource dependencies.

All auth integrations are optional and guarded: when the internal yijian
packages (``bceserver``/``bceidaas``/``yjsubscription``/``yjresource``) are not
installed, the auth dependencies are silently skipped so the server still
starts for local development.
"""
from __future__ import annotations

import os
from http import HTTPStatus

from fastapi import Depends, HTTPException, Request

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def make_default_model_dependency(server_context):
    """Build a dependency that fills a missing ``model`` field on POST JSON
    bodies with the engine's default model name."""

    async def add_default_model_name(request: Request):
        if request.method != 'POST':
            return None
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST,
                                detail={
                                    'error': {
                                        'message': 'Please request with json!',
                                        'type': 'invalid_request_error',
                                        'param': None,
                                        'code': 'invalid_json',
                                    }
                                })
        if not isinstance(body, dict):
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST,
                                detail={
                                    'error': {
                                        'message': 'Please request with json object!',
                                        'type': 'invalid_request_error',
                                        'param': None,
                                        'code': 'invalid_json_object',
                                    }
                                })
        # mutate the cached request json so the body model picks up the default
        if not body.get('model'):
            body['model'] = server_context.async_engine.model_name
        return body

    return add_default_model_name


def _build_auth_dependencies() -> list:
    """Build the optional BCE auth/subscription/resource dependencies."""
    depends: list = []
    try:
        from bceserver.middleware.impersonate import get_impersonate_dependency
        depends.append(Depends(get_impersonate_dependency()))

        from bceidaas.bce_client_configuration import BceClientConfiguration

        subs_endpoint = os.environ.get('SUBSCRIPTION_ENDPOINT', '')
        equity_id = os.environ.get('SUBSCRIPTION_EQUITY_ID', 'Endpoint/Multimodal/Count')
        if subs_endpoint:
            from subscriptionv1.middleware import get_subscription_dependency
            depends.append(Depends(get_subscription_dependency(BceClientConfiguration(endpoint=subs_endpoint),
                                                               equity_id)))

        resource_endpoint = os.environ.get('RESOURCE_ENDPOINT', '')
        metrics = os.environ.get('RESOURCE_METRICS', 'multimodal_count')
        if resource_endpoint:
            from resourcev1.middleware import get_resource_dependency
            depends.append(Depends(get_resource_dependency(BceClientConfiguration(endpoint=resource_endpoint),
                                                           metrics)))
    except ImportError as e:
        logger.warning(f'yijian auth packages are not available, skip auth dependencies: {e}')
    return depends


def build_openai_dependencies(server_context) -> list:
    """Assemble router-level dependencies for the OpenAI endpoints: optional
    auth (enabled via ``AUTH_ENABLED=true``) plus default model-name filling."""
    depends: list = []
    if os.environ.get('AUTH_ENABLED', 'false').lower() == 'true':
        depends.extend(_build_auth_dependencies())
    depends.append(Depends(make_default_model_dependency(server_context)))
    return depends
