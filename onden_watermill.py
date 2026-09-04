"""Onden Watermill — Hokusai's mill, wired to the tempo and lit up.

*Watermill at Onden* sits in **Thirty-Six Views of Mount Fuji**, the same series
as the Great Wave, so the wheel belongs to this picture as much as the sea
does.  Fuji is in the corner here for the same reason it is in the corner of
the print.

Where :mod:`floating_world` is a landscape built from plates, this is a
**machine**, and it is built the way a machine is: there is one shaft, and
everything else is downstream of it.  The wheel's angle drives the paddles, the
water each bucket is carrying, the pit gear meshing at the hub, and -- the part
that matters musically -- *when the splashes happen*.  A bucket tips at the
bottom of its travel and throws its water into the stream; nothing schedules
that on a timer.  So the rhythm you see is the wheel's own rhythm, and because
the wheel is locked to the tempo the mill beats time with the music rather than
alongside it.

It is an **overshot** wheel, which is what makes the cycle legible: the chute
pours onto the buckets near the top, they carry the water down the near side,
and they dump it at the bottom.  Fill, carry, throw -- a complete visible cycle
per bucket, twelve of them, all locked to one number.

The gearing is here on purpose.  A mill drives its machinery through a pit
wheel and a lantern pinion, so a turning gear is millwright's vocabulary rather
than borrowed steampunk, and it meshes with a ring on the hub at a real ratio.

Audio reactivity:
  - bass    -> chute flow, splash weight, wheel torque
  - mid     -> current speed, reed sway
  - treble  -> spray, sparkle on the water
  - beat    -> lantern flash, extra spray off the rim
  - tempo   -> the wheel itself, via the VJ API when an app is running

Randomised per activation: neon hue, sky variant, where the wheel starts in its
turn, reed and turtle placement, and ``self.time``.

Config keys (under ``[effects.OndenWatermill]``):
    speed, zoom, reactivity, glitter, neon_gain, flow_gain
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect
from unicornviz.vj_api import VJApi

__version__ = '0.3.0'

TAU = 2.0 * math.pi
_BUCKETS = 12
_NSPLASH = 8
# Where the wheel sits, and how big it is. Shared by the shader and the CPU,
# which needs them to know where a bucket throws its water.
WHEEL_X = -0.16
WHEEL_Y = -0.04
WHEEL_R = 0.30
SURFACE = -0.30                 # the stream's surface
# A bucket dumps as it comes round the bottom of its travel.
DUMP_ANGLE = 4.36               # radians, about 250 degrees

_VERT = """
#version 330
in vec2 in_vert;
out vec2 v_uv;
void main() {
    v_uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

_FRAG = """
#version 330
uniform float iTime;
uniform vec2  iResolution;
uniform float iBass;
uniform float iMid;
uniform float iTreble;
uniform float iBeat;
uniform float iPunch;
uniform float iHue;
uniform float iZoom;
uniform float iGlitter;
uniform float iNeon;
uniform float iSpin;        // the shaft angle: everything here hangs off it
uniform float iFlow;        // how hard the chute is running, 0..1
uniform float iCurrent;     // stream scroll, wrapped
uniform float iWind;        // reed sway, wrapped
uniform float iSky;
uniform vec4  iSplash[8];   // xy, z life, w size

in  vec2 v_uv;
out vec4 fragColor;

#define PI  3.14159265
#define TAU 6.28318530
#define NB  12.0
#define WX  (-0.16)
#define WY  (-0.04)
#define WR  0.30
#define SURF (-0.30)

float hash21(vec2 p) {
    p = mod(p, 512.0);
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 x) {
    vec2 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i),                  hash21(i + vec2(1.0, 0.0)), f.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), f.x), f.y);
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }
mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
float sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
float sdSeg(vec2 p, vec2 a, vec2 b, float r) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h) - r;
}
// Key block, then the neon it throws.
void ink(inout vec3 col, float d, float w, vec3 line, vec3 glow, float g) {
    col = mix(col, line, smoothstep(w, 0.0, abs(d)));
    col += glow * exp(-abs(d) * 95.0) * g;
}

// How much water the bucket at world angle *phi* is carrying. Overshot: filled
// under the chute near the top, carried down, thrown out at the bottom. This
// one function is why the splashes land where they do.
float bucketWater(float phi) {
    float p = mod(phi, TAU);
    return smoothstep(1.85, 2.15, p) * (1.0 - smoothstep(4.20, 4.55, p));
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float tb = mod(iTime, 600.0);
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0) / max(iZoom * (1.0 - iPunch * 0.045), 0.05);

    vec3 neonA = palette(iHue);
    vec3 neonB = palette(iHue + 0.42);
    vec3 neonC = palette(iHue + 0.72);
    float gain = clamp(iNeon, 0.0, 3.0);
    vec3 wood = mix(vec3(0.34, 0.20, 0.11), neonC, 0.16);
    vec3 water = mix(vec3(0.06, 0.34, 0.52), neonB, 0.26);

    // ── the sheet, and a bokashi wash behind everything ────────────────────
    vec3 col = vec3(0.085, 0.075, 0.105);
    float fib = vnoise(vec2(p.x * 190.0, p.y * 26.0)) * 0.6 + vnoise(p * 70.0) * 0.4;
    col *= 0.80 + fib * 0.45;
    float up = clamp(p.y * 1.15 + 0.5, 0.0, 1.0);
    vec3 lo, hi;
    if (iSky < 0.34)      { lo = vec3(0.92, 0.42, 0.20); hi = vec3(0.05, 0.08, 0.28); }
    else if (iSky < 0.67) { lo = vec3(0.25, 0.62, 0.72); hi = vec3(0.03, 0.09, 0.30); }
    else                  { lo = vec3(0.72, 0.24, 0.52); hi = vec3(0.03, 0.05, 0.22); }
    vec3 sky = mix(lo, hi, pow(up, 0.85));
    sky = mix(sky, neonB, 0.16 + iMid * 0.12);
    float steps = floor(up * 7.0) / 7.0;
    sky = mix(sky, sky * (0.86 + steps * 0.30), 0.55);
    col = mix(col, sky, 0.88);

    // Fuji, small and far, the way it sits in the corner of the print.
    {
        vec2 q = p - vec2(0.62, 0.13);
        float h = 0.15;
        float t = clamp(q.y / h, 0.0, 1.0);
        float edge = 0.19 * pow(1.0 - t, 1.85);
        float d = max(abs(q.x) - edge, max(-q.y, q.y - h));
        vec3 rock = mix(vec3(0.11, 0.12, 0.28), neonC, 0.10);
        rock = mix(rock, vec3(0.92, 0.95, 1.0),
                   smoothstep(h * 0.62, h * 0.74, q.y) * 0.9);
        col = mix(col, rock, smoothstep(0.002, 0.0, d));
        col += neonA * exp(-abs(d) * 110.0) * 0.22 * gain;
    }

    // ── the stream ─────────────────────────────────────────────────────────
    // Hokusai draws water as combed strokes rather than as a surface, so the
    // current here is rows of hooked lines scrolling past.
    if (p.y < SURF + 0.02) {
        float depth = SURF - p.y;
        col = mix(col, water * (0.55 - depth * 0.35), 0.93);
        float row = floor(depth * 22.0);
        float rh = hash21(vec2(row, 3.0));
        float ph = iCurrent * (0.55 + rh * 0.9) + row * 0.8;
        float curl = sin(p.x * 8.0 + ph) + 0.42 * sin(p.x * 17.0 - ph * 1.3);
        float line = smoothstep(0.30, 0.0,
                                abs(fract(depth * 22.0) - 0.5 - curl * 0.16));
        col += mix(vec3(0.80, 0.95, 1.0), neonB, 0.35) * line
             * (0.16 + iMid * 0.30) * smoothstep(0.0, 0.05, depth);
        // Sparkle riding the surface.
        col += vec3(0.9, 0.98, 1.0)
             * step(0.9965, hash21(floor(vec2(p.x * 240.0,
                                              p.y * 240.0 + iCurrent * 9.0))))
             * (0.4 + iTreble * 1.1) * iGlitter;
        ink(col, p.y - SURF, 0.0035, vec3(0.03, 0.08, 0.16), neonB, 0.35 * gain);
    }

    // ── the bank, its reeds and the turtles ────────────────────────────────
    {
        // The shore rises out of the water rather than sitting on it as a
        // slab: the waterline slopes up from where the bank begins, and the
        // ground simply continues to the edge of the sheet below it.
        float shore = 0.34 + 0.03 * sin(p.y * 9.0);
        float bankTop = SURF + smoothstep(shore, shore + 0.30, p.x) * 0.13
                      + 0.014 * sin(p.x * 5.3 + 1.4);
        float inBank = smoothstep(0.004, 0.0, p.y - bankTop)
                     * smoothstep(shore - 0.06, shore + 0.11, p.x);
        vec3 earth = mix(vec3(0.34, 0.26, 0.16), neonC, 0.10);
        earth *= 0.75 + 0.45 * smoothstep(bankTop - 0.09, bankTop, p.y);
        col = mix(col, earth, inBank * 0.95);
        ink(col, p.y - bankTop, 0.0030, vec3(0.06, 0.04, 0.04), neonC,
            0.22 * gain * inBank);

        // Reeds, leaning on the same wind the blossom uses next door. Each
        // stalk tapers to a tip, so a clump reads as reeds and not as a comb.
        float lean = 0.14 * sin(iWind * 0.05) + 0.06 * sin(iWind * 0.11 + p.y * 3.0);
        float rx = p.x + (p.y - bankTop) * lean;
        float col_ = floor(rx * 20.0);
        float rh = hash21(vec2(col_, 7.0));
        if (rh > 0.30 && p.x > shore + 0.03) {
            float tall = 0.10 + rh * 0.13;
            float t = clamp((p.y - bankTop) / tall, 0.0, 1.0);
            float w = 0.011 * (1.0 - t * 0.92);
            float reed = smoothstep(w, w * 0.3, abs(fract(rx * 20.0) - 0.5) / 20.0)
                       * step(bankTop - 0.005, p.y) * step(p.y, bankTop + tall);
            vec3 rc = mix(vec3(0.20, 0.46, 0.22), neonC, 0.26);
            col = mix(col, rc * (0.7 + t * 0.6), reed * 0.92);
            col += neonC * reed * 0.22 * gain * (0.4 + iMid * 0.7);
        }
        // Turtles on the rock -- Hokusai put them in, so they are in.
        for (int i = 0; i < 3; i++) {
            float fi = float(i);
            float tx = 0.46 + fi * 0.115;
            float ty = SURF + smoothstep(shore, shore + 0.30, tx) * 0.13 + 0.020;
            vec2 q = p - vec2(tx, ty + sin(tb * 0.7 + fi * 2.1) * 0.004);
            if (dot(q, q) > 0.006) continue;
            float shell = length(q * vec2(1.0, 1.55)) - 0.032;
            float head = length(q - vec2(0.038, -0.004)) - 0.012;
            float feet = min(length(q - vec2(0.020, -0.020)) - 0.009,
                             length(q - vec2(-0.020, -0.020)) - 0.009);
            float d = min(shell, min(head, feet));
            col = mix(col, mix(vec3(0.26, 0.40, 0.20), neonA, 0.16),
                      smoothstep(0.0025, 0.0, d));
            // The carved plates on the carapace.
            col = mix(col, vec3(0.06, 0.11, 0.07),
                      smoothstep(0.0028, 0.0, abs(shell + 0.011)) * 0.8);
            col = mix(col, vec3(0.06, 0.11, 0.07),
                      smoothstep(0.0026, 0.0, abs(d)) * 0.85);
            col += neonA * exp(-max(d, 0.0) * 120.0) * 0.22 * gain;
        }
    }

    // ── the chute ──────────────────────────────────────────────────────────
    // A wooden flume from off the top-left, pouring onto the buckets.
    {
        vec2 a0 = vec2(-0.95, 0.50), a1 = vec2(WX - 0.14, WY + WR + 0.15);
        float body = sdSeg(p, a0, a1, 0.028);
        col = mix(col, wood * 0.9, smoothstep(0.003, 0.0, body));
        ink(col, body, 0.0032, vec3(0.05, 0.03, 0.03), neonC, 0.30 * gain);
        // The pour: a ribbon of water leaving the lip and falling onto the rim.
        // Parameterised down the fall, so the ribbon is continuous from the
        // lip to the rim instead of a blob at one end.
        // The fall runs from the lip to the rim and stops there, rather than
        // carrying on through the wheel.
        vec2 lip = a1 + vec2(0.010, -0.006);
        float fallBot = WY + WR - 0.015;
        float t = clamp((lip.y - p.y) / max(lip.y - fallBot, 1e-3), 0.0, 1.0);
        float px = lip.x + t * 0.085;
        float wob = sin(p.y * 55.0 - iCurrent * 6.0) * 0.004 * iFlow;
        float pour = smoothstep(0.018 * (0.55 + iFlow * 0.9), 0.0,
                                abs(p.x - px - wob))
                   * step(fallBot, p.y) * step(p.y, lip.y);
        col = mix(col, mix(water, vec3(0.85, 0.97, 1.0), 0.5), pour * iFlow);
        col += neonB * pour * iFlow * 0.55 * gain;
    }

    // ── the wheel ──────────────────────────────────────────────────────────
    {
        vec2 q = p - vec2(WX, WY);
        float r = length(q);
        float phi = atan(q.y, q.x);
        // Angle in the wheel's own frame, so the paddles turn with the shaft.
        float local = phi - iSpin;
        float cellF = fract(local / TAU * NB);
        float cellA = (cellF - 0.5) * TAU / NB;      // radians off this paddle
        // The paddle: a radial plank out at the rim, with a bucket floor.
        float plank = max(abs(cellA) * r - 0.010, max(WR * 0.74 - r, r - WR * 1.02));
        float floorD = max(abs(r - WR * 0.80) - 0.008,
                           abs(cellA) * r - (TAU / NB) * WR * 0.42);
        float paddle = min(plank, floorD);

        // What this bucket is carrying, from its own world angle.
        float bwc = (floor(local / TAU * NB) + 0.5) * TAU / NB + iSpin;
        float carry = bucketWater(bwc);
        float pocket = max(max(WR * 0.82 - r, r - WR * 0.99),
                           abs(cellA) * r - (TAU / NB) * WR * 0.36);
        col = mix(col, mix(water, vec3(0.75, 0.95, 1.0), 0.35),
                  smoothstep(0.003, 0.0, pocket) * carry);
        col += neonB * smoothstep(0.003, 0.0, pocket) * carry * 0.45 * gain;

        // Rim tyres, spokes and hub.
        float tyre = min(abs(r - WR), abs(r - WR * 0.74)) - 0.006;
        float spoke = max(abs(sin(local * 3.0)) * r - 0.009, r - WR * 0.76);
        float hub = abs(r - WR * 0.10) - 0.012;
        float frame = min(min(tyre, spoke), hub);
        col = mix(col, wood, smoothstep(0.003, 0.0, min(frame, paddle)));
        ink(col, min(frame, paddle), 0.0030, vec3(0.05, 0.03, 0.04), neonC,
            0.34 * gain * (0.5 + iBass * 0.6));

        // Water sheeting off the rim as it comes out of the stream.
        float wet = smoothstep(0.0, -0.25, phi) * smoothstep(-PI, -PI + 0.6, phi);
        col += mix(water, vec3(0.9, 1.0, 1.0), 0.4)
             * smoothstep(0.02, 0.0, abs(r - WR)) * wet
             * (0.35 + iTreble * 0.8) * gain;
    }

    // ── the gearing ────────────────────────────────────────────────────────
    // A pit wheel on the shaft driving a smaller pinion: a mill's own
    // vocabulary, and it turns at the ratio the tooth counts demand.
    {
        vec2 gc = vec2(WX + 0.30, WY - 0.13);
        vec2 q = p - gc;
        float r = length(q);
        if (r < 0.115) {
            float ratio = -NB / 7.0;            // driven, so it counter-turns
            float a = atan(q.y, q.x) - iSpin * ratio;
            float R = 0.085;
            float teeth = R * (1.0 + 0.10 * step(0.5, fract(a / TAU * 7.0)));
            float rim = smoothstep(0.009, 0.0, abs(r - teeth));
            float hub = smoothstep(0.008, 0.0, abs(r - R * 0.24));
            float arm = smoothstep(0.007, 0.0, abs(sin(a * 2.0)) * r)
                      * step(r, R * 0.92);
            float m = clamp(rim + hub + arm + smoothstep(0.003, 0.0, r - teeth) * 0.25,
                            0.0, 1.0);
            col = mix(col, mix(vec3(0.62, 0.44, 0.18), neonC, 0.15)
                           * (0.7 + iMid * 0.4), m);
            col += neonC * m * 0.22 * gain * (0.35 + iBass * 0.6);
        }
    }

    // ── the axle trestle ───────────────────────────────────────────────────
    {
        vec2 c = vec2(WX, WY);
        float leg = min(sdSeg(p, c, vec2(WX - 0.20, SURF - 0.10), 0.014),
                        sdSeg(p, c, vec2(WX + 0.20, SURF - 0.10), 0.014));
        float brace = sdSeg(p, vec2(WX - 0.13, WY - 0.16),
                            vec2(WX + 0.13, WY - 0.16), 0.010);
        float d = min(leg, brace);
        col = mix(col, wood * 0.75, smoothstep(0.003, 0.0, d));
        ink(col, d, 0.0028, vec3(0.05, 0.03, 0.04), neonC, 0.22 * gain);
    }

    // ── what the buckets throw ─────────────────────────────────────────────
    for (int i = 0; i < 8; i++) {
        vec4 S = iSplash[i];
        if (S.z <= 0.0) continue;
        vec2 d = p - S.xy;
        float q2 = dot(d, d);
        if (q2 > 0.02) continue;
        // A crown of droplets thrown up out of the surface.
        float ang = atan(d.y, d.x);
        float rr = length(d);
        float crown = smoothstep(S.w * (0.4 + S.z * 0.9), 0.0, abs(rr - S.w * S.z))
                    * step(0.0, d.y)
                    * (0.55 + 0.45 * sin(ang * 11.0 + S.x * 30.0));
        col += mix(vec3(0.88, 0.97, 1.0), neonB, 0.25) * crown * S.z
             * (0.7 + iTreble * 0.8) * gain;
        // A tight bloom at the point of entry, not a dome over the whole area.
        col += mix(water, vec3(1.0), 0.6) * exp(-q2 * 5200.0) * S.z * 0.45;
    }

    // ── lanterns on the mill ───────────────────────────────────────────────
    for (int i = 0; i < 2; i++) {
        vec2 lc = vec2(-0.62 + float(i) * 0.16, 0.20 - float(i) * 0.06);
        float d = length((p - lc) * vec2(1.0, 0.72)) - 0.030;
        vec3 lamp = mix(vec3(1.0, 0.62, 0.24), neonA, 0.25);
        col = mix(col, lamp * (0.8 + iBeat * 0.8), smoothstep(0.003, 0.0, d));
        col = mix(col, vec3(0.06, 0.03, 0.03),
                  smoothstep(0.002, 0.0, abs(d + 0.012)) * 0.5);
        col += lamp * exp(-max(d, 0.0) * 26.0) * (0.35 + iBeat * 0.9) * gain;
    }

    // ── the seal, and the printed edge ─────────────────────────────────────
    vec2 sq = p - vec2(A * 0.5 - 0.085, -0.5 + 0.085);
    float sbox = sdBox(sq, vec2(0.036, 0.036));
    float carve = min(min(sdBox(sq - vec2(0.0,  0.014), vec2(0.020, 0.004)),
                          sdBox(sq - vec2(0.0, -0.006), vec2(0.020, 0.004))),
                      sdBox(sq - vec2(0.0, -0.024), vec2(0.020, 0.004)));
    carve = min(carve, sdBox(sq - vec2(-0.020, 0.0), vec2(0.004, 0.030)));
    vec3 seal = mix(vec3(0.86, 0.12, 0.10), neonA, 0.25);
    col = mix(col, seal * (0.7 + iBeat * 0.9),
              smoothstep(0.003, 0.0, sbox) * smoothstep(0.0, 0.003, carve));
    col += seal * exp(-abs(sbox) * 60.0) * (0.25 + iBeat * 0.8) * gain;

    float border = sdBox(p, vec2(A * 0.5 - 0.020, 0.5 - 0.020));
    col = mix(col, vec3(0.05, 0.05, 0.08), smoothstep(0.0, 0.003, border));
    col += neonB * exp(-abs(border) * 120.0) * 0.30 * gain;

    col += neonA * iBeat * 0.05;
    col *= smoothstep(1.45, 0.30, length(v_uv - 0.5));
    col = col / (1.0 + col * 0.42);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.86));
    fragColor = vec4(col, 1.0);
}
"""


class OndenWatermill(BaseEffect):
    """Hokusai's overshot mill, its shaft locked to the tempo."""

    NAME = 'Onden Watermill'
    AUTHOR = 'unicorn-viz'
    TAGS = ['ukiyo-e', 'neon', 'steampunk', 'groovy', 'chill', 'showcase']
    PING_PONG_FRIENDS = ['Floating World', 'Audio Spectrum']

    # The wheel turns a whole bucket per beat, so a paddle tips at the bottom
    # on the beat rather than merely near it.
    _BUCKETS_PER_BEAT = 1.0

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'zoom': float(self.config.get('zoom', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.3)),
            'glitter': float(self.config.get('glitter', 1.0)),
            'neon_gain': float(self.config.get('neon_gain', 1.0)),
            'flow_gain': float(self.config.get('flow_gain', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._punch = 0.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._sky = float(self.rng.uniform(0.0, 1.0))
        self._spin = float(self.rng.uniform(0.0, TAU))
        self._flow = 0.6
        self._current = float(self.rng.uniform(0.0, 100.0))
        self._wind = float(self.rng.uniform(0.0, 100.0))
        self._bpm = 120.0
        self._vj_poll = 0.0
        # Which bucket last tipped, so a dump is detected once and only once.
        self._last_dump = self._dump_index(self._spin)
        self._splashes = np.zeros((_NSPLASH, 4), dtype=np.float64)
        self._tbl_splash = np.zeros((_NSPLASH, 4), dtype='f4')

    @staticmethod
    def _dump_index(spin: float) -> int:
        """Which bucket has most recently passed the dump angle.

        The whole scene's rhythm comes off this: a bucket crossing the angle is
        what fires a splash, so the splashes are the mechanism's, not a timer's.
        """
        return int(math.floor((spin - DUMP_ANGLE) / TAU * _BUCKETS))

    def _read_runtime(self, dt: float) -> None:
        """Take the tempo from the VJ API when an app is running to give it."""
        self._vj_poll -= dt
        if self._vj_poll > 0.0:
            return
        self._vj_poll = 0.25
        api = VJApi.current()
        if api is None:
            return
        try:
            bpm = api.get_bpm()
        except Exception:            # noqa: BLE001 - context is never required
            return
        if bpm and bpm > 20.0:
            self._bpm = float(bpm)

    def _throw_water(self) -> None:
        """A bucket has tipped: put its water in the stream."""
        free = np.where(self._splashes[:, 2] <= 0.0)[0]
        slot = int(free[0]) if free.size else int(np.argmin(self._splashes[:, 2]))
        # It lands where the bucket actually is, below the wheel's near side.
        x = WHEEL_X + math.cos(DUMP_ANGLE) * WHEEL_R
        self._splashes[slot] = (
            x + float(self.rng.uniform(-0.02, 0.02)),
            SURFACE,
            1.0,
            0.038 + self._bass * 0.032 + float(self.rng.uniform(0.0, 0.010)),
        )

    def update(self, dt: float, audio: AudioData) -> None:
        """Turn the shaft, and let everything downstream of it follow."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 2.0))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 2.0))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 2.0))
        if audio.beat > 0.5:
            self._beat = 1.0
            self._punch = 1.0
        self._beat = max(0.0, self._beat - dt * 4.0)
        self._punch = max(0.0, self._punch - dt * 2.4)

        speed = max(0.05, float(self.parameters['speed']))
        step = dt * speed
        self._read_runtime(dt)

        # The shaft. One bucket per beat at the reported tempo, so the mill
        # keeps time with the music instead of near it; the low end leans on
        # the wheel a little, the way more water on the buckets would.
        rate = (self._bpm / 60.0) * self._BUCKETS_PER_BEAT * (TAU / _BUCKETS)
        self._spin = (self._spin
                      + step * rate * (0.85 + self._bass * 0.30)) % TAU

        # A bucket crossing the dump angle throws its water. Compared against
        # the previous frame's index, so a dump fires once however fast the
        # wheel is turning and never twice on a slow frame.
        idx = self._dump_index(self._spin)
        if idx != self._last_dump:
            self._throw_water()
            self._last_dump = idx

        flow_gain = max(0.0, float(self.parameters['flow_gain']))
        self._flow = float(np.clip(
            (0.45 + self._bass * 0.55) * flow_gain, 0.0, 1.5))
        self._current = (self._current
                         + step * (0.7 + self._mid * 1.3)) % 100.0
        self._wind = (self._wind + step * (0.5 + self._mid * 0.9)) % 100.0

        for i in range(_NSPLASH):
            s = self._splashes[i]
            if s[2] <= 0.0:
                continue
            s[2] -= step * 1.5
            s[3] += step * 0.035

    def render(self) -> None:
        """Upload the shaft angle and everything hanging off it."""
        p = self._prog
        p['iTime'].value = float(self.time)
        p['iResolution'].value = (float(self.width), float(self.height))
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iPunch'].value = self._punch
        p['iHue'].value = self._hue
        p['iZoom'].value = max(0.05, float(self.parameters['zoom']))
        p['iGlitter'].value = max(0.0, float(self.parameters['glitter']))
        p['iNeon'].value = max(0.0, float(self.parameters['neon_gain']))
        p['iSpin'].value = float(self._spin)
        p['iFlow'].value = float(self._flow)
        p['iCurrent'].value = float(self._current)
        p['iWind'].value = float(self._wind)
        p['iSky'].value = self._sky
        self._tbl_splash[:] = self._splashes
        self._tbl_splash[:, 2] = np.clip(self._splashes[:, 2], 0.0, 1.0)
        p['iSplash'].write(self._tbl_splash.tobytes())
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
