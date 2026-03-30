from __future__ import annotations

import logging
from typing import Any, List, Optional

from omegaconf import DictConfig, ListConfig

from curator.model.conversion import convert_single_to_multi_domain
from curator.model.multi_domain import MultiDomainPotential, apply_domain_set


def prepare_multi_domain_finetune(
    config: DictConfig,
    datamodule: Any,
    model,
    logger: Optional[logging.Logger] = None,
):
    if not getattr(datamodule, "domain_modules", None):
        raise ValueError("Multi-domain fine-tune requires a datamodule with domain_modules.")

    task_cfg = getattr(config, "task", None)
    mode = str(getattr(task_cfg, "domain_mode", "") or "").strip().lower()
    if mode not in {"extend", "replace"}:
        mode = "extend"

    def normalize_domains(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (ListConfig, list, tuple, set)):
            return [str(item) for item in value]
        return [str(value)]

    def unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(str(value) for value in values))

    explicit_domains = unique(normalize_domains(getattr(task_cfg, "new_domains", None)))
    if explicit_domains:
        domains = explicit_domains
    else:
        domain_to_id = getattr(datamodule, "domain_to_id", {}) or {}
        domains = unique(
            [
                str(domain_to_id.get(name, name))
                for name in getattr(datamodule, "domain_modules", {}).keys()
                if not str(name).lower().startswith("replay")
            ]
        )
        if logger and domains:
            logger.debug("Inferred multi-domain target domains=%s from datamodule.", domains)

    template_domain = str(getattr(task_cfg, "init_new_domains_from", None) or "0")
    init_strategy = "copy" if getattr(task_cfg, "init_new_domains_from", None) is not None else "random"

    if not isinstance(model, MultiDomainPotential):
        model = convert_single_to_multi_domain(model)
        if logger:
            logger.debug("Wrapped model with explicit multi-domain structure.")

    updated = apply_domain_set(
        model,
        domains,
        mode=mode,
        template_domain=template_domain,
        init_strategy=init_strategy,
        logger=logger,
    )
    if logger:
        logger.debug(
            "Prepared multi-domain model: mode=%s domains=%s updated_modules=%s",
            mode,
            domains,
            updated,
        )
    return model


__all__ = ["prepare_multi_domain_finetune"]
