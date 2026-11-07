# Regression: the VW gap button stalk reports GRA_Verstellung_Zeitluecke / GRA_Zeitluecke as an
# ACTION code (per MEB value table: 1=dist minus, 2=dist plus, 3=toggle), not a momentary 1.
# Mapping gapAdjustCruise to only [1] made every "increase distance"/"toggle" press invisible,
# so carrot's short-press gap cycle and long-press MyDrivingMode switch never fired on MQB/PQ.
import pytest

from opendbc.car import structs
from opendbc.car.volkswagen.carstate import CarState
from opendbc.car.volkswagen.values import CAR, VolkswagenFlags

ButtonType = structs.CarState.ButtonEvent.Type


class FakeParser:
  def __init__(self, msg: str, signals: list[str]):
    self.vl = {msg: {s: 0 for s in signals}}


MQB_GRA_SIGNALS = ["GRA_Tip_Setzen", "GRA_Tip_Wiederaufnahme", "GRA_Tip_Hoch", "GRA_Tip_Runter",
                   "GRA_Abbrechen", "GRA_Verstellung_Zeitluecke", "GRA_Typ_Hauptschalter", "GRA_Tip_Stufe_2"]
PQ_GRA_SIGNALS = ["GRA_Neu_Setzen", "GRA_Recall", "GRA_Up_kurz", "GRA_Down_kurz",
                  "GRA_Abbrechen", "GRA_Zeitluecke"]


def _mqb_carstate():
  CP = structs.CarParams()
  CP.carFingerprint = CAR.VOLKSWAGEN_GOLF_MK7.value
  CP.flags = 0
  return CarState(CP)


def _edges(cs, pt, sig: str, values: list[int]):
  out = []
  for v in values:
    pt.vl[next(iter(pt.vl))][sig] = v
    for b in cs.create_button_events(pt, cs.CCP.BUTTONS):
      out.append((b.type, b.pressed))
  return out


@pytest.mark.parametrize("action_value", [1, 2, 3], ids=["dist_minus", "dist_plus", "toggle"])
def test_mqb_gap_button_fires_on_every_stalk_action(action_value):
  cs = _mqb_carstate()
  pt = FakeParser("GRA_ACC_01", MQB_GRA_SIGNALS)
  edges = _edges(cs, pt, "GRA_Verstellung_Zeitluecke", [0, action_value, 0])
  assert edges == [(ButtonType.gapAdjustCruise, True), (ButtonType.gapAdjustCruise, False)]


@pytest.mark.parametrize("action_value", [1, 2, 3], ids=["dist_minus", "dist_plus", "toggle"])
def test_pq_gap_button_fires_on_every_stalk_action(action_value):
  CP = structs.CarParams()
  CP.carFingerprint = CAR.VOLKSWAGEN_CADDY_MK3.value
  CP.flags = int(VolkswagenFlags.PQ)
  cs = CarState(CP)
  pt = FakeParser("GRA_Neu", PQ_GRA_SIGNALS)
  out = []
  for v in [0, action_value, 0]:
    pt.vl["GRA_Neu"]["GRA_Zeitluecke"] = v
    for b in cs.create_button_events(pt, cs.CCP.BUTTONS):
      out.append((b.type, b.pressed))
  assert out == [(ButtonType.gapAdjustCruise, True), (ButtonType.gapAdjustCruise, False)]


def test_mqb_speed_rocker_still_separate():
  # pressing +/- must not be swallowed as gap events
  cs = _mqb_carstate()
  pt = FakeParser("GRA_ACC_01", MQB_GRA_SIGNALS)
  edges = _edges(cs, pt, "GRA_Tip_Hoch", [1, 0])
  assert (ButtonType.gapAdjustCruise, True) not in edges
  assert (ButtonType.accelCruise, True) in edges
