from __future__ import annotations

import json
import os
import random
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth


@dataclass(frozen=True)
class ColabFoldMSAConfig:
    host_url: str = "https://api.colabfold.com"
    use_env: bool = True
    use_filter: bool = True
    timeout_sec: float = 30.0
    poll_min_sec: float = 5.0
    poll_jitter_sec: float = 5.0
    max_submit_retries: int = 5
    user_agent: str = "protenixscore"


def _a3m_first_sequence(a3m_text: str) -> Optional[str]:
    for line in a3m_text.splitlines():
        if not line:
            continue
        if line.startswith(">"):
            continue
        return line.strip().replace("-", "")
    return None


def _build_colabfold_mode(use_env: bool, use_filter: bool) -> str:
    if use_filter:
        return "env" if use_env else "all"
    return "env-nofilter" if use_env else "nofilter"


def _fetch_colabfold_unpaired_a3m(
    *,
    sequence: str,
    work_dir: Path,
    cfg: ColabFoldMSAConfig,
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    auth_headers: Optional[dict[str, str]] = None,
) -> str:
    has_basic_auth = bool(msa_server_username and msa_server_password)
    has_header_auth = auth_headers is not None
    if has_basic_auth and has_header_auth:
        raise ValueError("Cannot use both basic and header auth.")

    headers: dict[str, str] = {"User-Agent": cfg.user_agent}
    auth = None
    if has_basic_auth:
        auth = HTTPBasicAuth(msa_server_username, msa_server_password)
    if has_header_auth:
        headers.update(auth_headers or {})

    mode = _build_colabfold_mode(cfg.use_env, cfg.use_filter)
    submission_endpoint = "ticket/msa"

    query_id = 101
    query = f">{query_id}\n{sequence}\n"

    def _post_ticket() -> dict:
        res = requests.post(
            f"{cfg.host_url}/{submission_endpoint}",
            data={"q": query, "mode": mode},
            timeout=cfg.timeout_sec,
            headers=headers,
            auth=auth,
        )
        try:
            return res.json()
        except Exception:
            raise RuntimeError(
                f"ColabFold server did not return JSON (status={res.status_code}): {res.text[:500]}"
            )

    def _get_status(ticket_id: str) -> dict:
        res = requests.get(
            f"{cfg.host_url}/ticket/{ticket_id}",
            timeout=cfg.timeout_sec,
            headers=headers,
            auth=auth,
        )
        try:
            return res.json()
        except Exception:
            raise RuntimeError(
                f"ColabFold server did not return JSON (status={res.status_code}): {res.text[:500]}"
            )

    def _download(ticket_id: str, dst: Path) -> None:
        res = requests.get(
            f"{cfg.host_url}/result/download/{ticket_id}",
            timeout=cfg.timeout_sec,
            headers=headers,
            auth=auth,
        )
        res.raise_for_status()
        dst.write_bytes(res.content)

    last_err: Optional[Exception] = None
    out: dict | None = None
    for _ in range(cfg.max_submit_retries + 1):
        try:
            out = _post_ticket()
            last_err = None
            break
        except Exception as e:
            last_err = e
            time.sleep(5)
    if out is None:
        raise RuntimeError("Failed to submit ColabFold MSA request.") from last_err

    while out.get("status") in {"UNKNOWN", "RATELIMIT"}:
        time.sleep(cfg.poll_min_sec + random.random() * cfg.poll_jitter_sec)
        out = _post_ticket()

    if out.get("status") in {"ERROR", "MAINTENANCE"}:
        raise RuntimeError(
            f"ColabFold MSA submission failed: status={out.get('status')} payload={json.dumps(out)[:500]}"
        )

    ticket_id = out.get("id")
    if not ticket_id:
        raise RuntimeError(f"ColabFold response missing ticket id: {out}")

    while out.get("status") in {"UNKNOWN", "RUNNING", "PENDING"}:
        time.sleep(cfg.poll_min_sec + random.random() * cfg.poll_jitter_sec)
        out = _get_status(ticket_id)

    if out.get("status") != "COMPLETE":
        raise RuntimeError(
            f"ColabFold MSA job did not complete: status={out.get('status')} payload={json.dumps(out)[:500]}"
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    tar_gz = work_dir / f"colabfold_{mode}.tar.gz"
    extract_dir = work_dir / f"colabfold_{mode}"
    if not tar_gz.exists():
        _download(ticket_id, tar_gz)

    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_gz) as tf:
        tf.extractall(extract_dir)

    a3m_files = [extract_dir / "uniref.a3m"]
    if cfg.use_env:
        a3m_files.append(extract_dir / "bfd.mgnify30.metaeuk30.smag30.a3m")

    missing = [str(p) for p in a3m_files if not p.exists()]
    if missing:
        raise RuntimeError(
            f"ColabFold download missing expected A3M files: {missing}. "
            f"Directory contents: {[p.name for p in extract_dir.iterdir()]}"
        )

    a3m_lines: list[str] = []
    for p in a3m_files:
        update_m = True
        m = None
        for line in p.read_text().splitlines(keepends=True):
            if "\x00" in line:
                line = line.replace("\x00", "")
                update_m = True
            if line.startswith(">") and update_m:
                try:
                    m = int(line[1:].rstrip())
                except Exception:
                    m = None
                update_m = False
            if m == query_id:
                a3m_lines.append(line)

    if not a3m_lines:
        raise RuntimeError("ColabFold produced empty A3M for the query sequence.")
    return "".join(a3m_lines)


def ensure_msa_dir(
    *,
    sequence: str,
    out_dir: Path,
    cfg: ColabFoldMSAConfig,
    force: bool = False,
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    auth_headers: Optional[dict[str, str]] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    non_pairing = out_dir / "non_pairing.a3m"
    pairing = out_dir / "pairing.a3m"

    if not force and non_pairing.exists() and pairing.exists():
        cached_seq = _a3m_first_sequence(non_pairing.read_text())
        if cached_seq == sequence:
            return out_dir

    msa_text = _fetch_colabfold_unpaired_a3m(
        sequence=sequence,
        work_dir=out_dir / "_colabfold_tmp",
        cfg=cfg,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        auth_headers=auth_headers,
    )

    non_pairing.write_text(msa_text)
    pairing.write_text(msa_text)

    tmp_dir = out_dir / "_colabfold_tmp"
    try:
        if tmp_dir.exists():
            for root, dirs, files in os.walk(tmp_dir, topdown=False):
                for fn in files:
                    Path(root, fn).unlink(missing_ok=True)
                for dn in dirs:
                    Path(root, dn).rmdir()
            tmp_dir.rmdir()
    except Exception:
        pass

    return out_dir
