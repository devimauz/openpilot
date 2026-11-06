"""carrot_navi v2 feed client: control handshake + per-item JSON websockets (asyncio)."""
from __future__ import annotations

import asyncio
import json
import time

import websockets

from tomtom2carrot import PROTOCOL_VERSION, JSON_NAMES, requirements_query


def now_ms() -> int:
  return time.time_ns() // 1_000_000


class NaviFeedClient:
  def __init__(self, host: str, port: int = 7714, app_version: str = "tomtom-bridge-1.0") -> None:
    self.host, self.port, self.app_version = host, port, app_version
    self.session_id: str | None = None
    self._seq: dict[str, int] = {}
    self._socks: dict[str, any] = {}
    self.last_sent: dict[str, dict] = {}

  def _control_url(self) -> str:
    return f"ws://{self.host}:{self.port}/api/navi/ws/v2/control/{self.app_version}"

  def _item_url(self, name: str) -> str:
    return f"ws://{self.host}:{self.port}/api/navi/ws/v2/json/{self.session_id}/{name}"

  async def connect(self) -> dict:
    self.ws_control = await websockets.connect(self._control_url(), open_timeout=5, max_size=2**23)
    await self.ws_control.send(json.dumps(requirements_query()))
    manifest = json.loads(await asyncio.wait_for(self.ws_control.recv(), timeout=5))
    if manifest.get("type") != "subscription_manifest":
      raise RuntimeError(f"bad handshake reply: {manifest}")
    self.session_id = manifest["session_id"]
    for name in JSON_NAMES:
      sock = await websockets.connect(self._item_url(name), open_timeout=5, max_size=2**23)
      self._socks[name] = sock
      self._seq[name] = 0
    return manifest

  def _envelope(self, name: str, item: dict) -> dict:
    self._seq[name] += 1
    ms = now_ms()
    env = {"type": "item_update", "protocol_version": PROTOCOL_VERSION,
           "session_id": self.session_id, "kind": "json", "name": name,
           "sequence": self._seq[name], "source_timestamp_ms": ms, "sent_at_ms": ms}
    if item.get("present"):
      env["present"] = True
      env["value"] = item["value"]
    else:
      env["present"] = False
      env["value"] = None
      env["reason"] = item.get("reason", "source_absent")
    return env

  async def push_items(self, items: dict[str, dict], force: bool = False) -> int:
    sent = 0
    for name, item in items.items():
      sock = self._socks.get(name)
      if sock is None:
        continue
      if not force and self.last_sent.get(name) == item:
        continue
      await sock.send(json.dumps(self._envelope(name, item)))
      self.last_sent[name] = json.loads(json.dumps(item))
      sent += 1
    return sent

  async def close(self) -> None:
    for sock in self._socks.values():
      await sock.close()
    self._socks.clear()
    if getattr(self, "ws_control", None):
      await self.ws_control.close()


async def probe(host: str, ticks: int = 10, period: float = 1.0) -> None:
  client = NaviFeedClient(host)
  manifest = await client.connect()
  print(f"session={manifest['session_id']} streams={len(manifest['streams'])}")
  from feed_sim import scenario_snapshot
  try:
    for t in range(0, ticks * 3, 3):
      snap = scenario_snapshot(t)
      n = await client.push_items(snap.build_items())
      print(f"t={t}s pushed {n} items")
      await asyncio.sleep(period)
  finally:
    await client.close()


if __name__ == "__main__":
  import argparse
  ap = argparse.ArgumentParser()
  ap.add_argument("--host", required=True)
  ap.add_argument("--ticks", type=int, default=10)
  a = ap.parse_args()
  asyncio.run(probe(a.host, a.ticks))
