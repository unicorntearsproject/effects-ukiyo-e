"""Night Parade — Hyakki Yagyo, lit only by the lanterns it is carrying.

The *Hyakki Yagyo* handscrolls show the hundred demons crossing town at night
in single file, each with its own lantern.  The scroll unrolls, so the parade
is the composition: there is no frame, only a procession that keeps arriving.

Where :mod:`floating_world` is a landscape built from plates and
:mod:`onden_watermill` is a machine downstream of one shaft, this is **a
procession**, and everything hangs off a single number -- how far the parade
has walked.  The march position gives each yokai its place in the line; the
place gives it the phase of its own walk; the walk shakes the pole it is
carrying; the pole swings the lantern; and the lantern is a *light*.

That last step is what stops this being parallax layers:

  - **The lanterns are the only light source.**  The road, the mist, the wall
    behind and every yokai are lit by summing the lamps at their actual
    positions this frame.  Nothing has an ambient term worth speaking of, so
    when the parade moves, the lighting moves with it, and the far end of the
    street is dark because nothing is standing there yet.
  - **The shadows are cast, not drawn.**  A yokai's shadow on the wall is
    found by running the ray backwards: for a wall pixel, the point that would
    occlude it is ``lamp + (p - lamp) / spread``, and the shadow is just that
    point tested against the same silhouette the yokai is drawn from.  So a
    shadow leans when its own lantern swings, and a yokai walking past another
    one's lamp throws a second shadow the other way.

The lantern's swing is a real damped pendulum driven by the carrier's
sideways acceleration, so it lags the walk instead of matching it, and a hop
on the beat sets it going.

It is the 1830s vocabulary run through neon and brass: the chochin are
filament-lit, the wall has conduit in its plaster, the oni's club is riveted,
and the seal glows.

Audio reactivity:
  - bass    -> the pace of the march, and the lanterns' burn
  - mid     -> mist depth and the wall's conduit glow
  - treble  -> embers coming off the lamps
  - beat    -> the whole line hops, which sets every pendulum swinging

Randomised per activation: palette, which yokai are in the line and in what
order, their sizes, the march's direction, and the moon's phase and place.

Config keys (under ``[effects.NightParade]``):
    speed, reactivity, glitter, neon_gain
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.4.0'

_NYOKAI = 7
# Wide enough that the line has gaps in it. At 2.80 the seven of them stood
# shoulder to shoulder and the parade read as a crowd, which also left no dark
# street between the lamps for the lighting to show off against.
_SPAN = 3.15
_ROAD_Y = -0.340        # where the yokai's feet are
_TAU = 2.0 * math.pi

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
uniform vec2  iResolution;
uniform float iTime;
uniform float iBass;
uniform float iMid;
uniform float iTreble;
uniform float iBeat;
uniform float iHue;
uniform float iGlitter;
uniform float iNeon;
uniform float iMist;
uniform float iMarch;
uniform vec3  iMoon;        // xy, phase
uniform vec4  iYokai[7];    // x, kind, bob, scale
uniform vec4  iLamp[7];     // x, y, tilt, burn
in  vec2 v_uv;
out vec4 fragColor;

#define TAU 6.28318530
#define ROAD -0.340

float hash21(vec2 p) {
    p = mod(p, 512.0);
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1, 0)), f.x),
               mix(hash21(i + vec2(0, 1)), hash21(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) { s += a * vnoise(p); p *= 2.05; a *= 0.5; }
    return s;
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

float sdSeg(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
}
float sdBox(vec2 p, vec2 b, float r) {
    vec2 d = abs(p) - b + r;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - r;
}

// One silhouette per yokai, in a local frame with the feet at the origin and
// a nominal height of 1. Drawn from the same function the shadows are tested
// against, so a shadow can never disagree with the thing casting it.
float yokai(vec2 q, int kind, float bob) {
    q.y -= bob * 0.035;
    if (kind == 0) {
        // Karakasa: the one-legged umbrella. Canopy, shaft, single foot.
        float canopy = max(length(vec2(q.x, (q.y - 0.72) * 1.9)) - 0.30,
                           0.70 - q.y);
        float scallop = abs(q.y - 0.70 - sin(q.x * 26.0) * 0.014) - 0.014;
        canopy = min(canopy, max(scallop, abs(q.x) - 0.29));
        float shaft = sdSeg(q, vec2(0.0, 0.14), vec2(0.0, 0.86)) - 0.020;
        float leg = sdSeg(q, vec2(0.0, 0.16), vec2(0.03, 0.03)) - 0.028;
        float foot = sdBox(q - vec2(0.055, 0.022), vec2(0.075, 0.022), 0.016);
        float tongue = sdSeg(q, vec2(0.0, 0.58), vec2(0.02, 0.40)) - 0.038;
        return min(min(min(canopy, shaft), min(leg, foot)), tongue);
    }
    if (kind == 1) {
        // Chochin-obake: the lantern ghost -- ribbed body, split grin.
        float body = sdBox(q - vec2(0.0, 0.52), vec2(0.20, 0.30), 0.14);
        float cap = sdBox(q - vec2(0.0, 0.84), vec2(0.13, 0.030), 0.012);
        float base = sdBox(q - vec2(0.0, 0.19), vec2(0.13, 0.030), 0.012);
        float arms = min(sdSeg(q, vec2(-0.18, 0.55), vec2(-0.34, 0.40)),
                         sdSeg(q, vec2(0.18, 0.55), vec2(0.34, 0.40))) - 0.020;
        // The split, and the tongue lolling out of it -- without those it is
        // a rounded box and nothing else.
        float split = max(abs(q.x) - 0.021, abs(q.y - 0.50) - 0.19);
        float tongue = sdSeg(q, vec2(0.0, 0.44), vec2(0.055, 0.31)) - 0.035;
        return min(min(max(min(min(body, arms), min(cap, base)), -split), tongue),
                   1e9);
    }
    if (kind == 2) {
        // Oni: squat, horned, with a club over the shoulder.
        float torso = sdBox(q - vec2(0.0, 0.42), vec2(0.19, 0.20), 0.10);
        float head = length((q - vec2(0.0, 0.75)) * vec2(1.0, 1.08)) - 0.155;
        float horn = min(sdSeg(q, vec2(-0.085, 0.85), vec2(-0.175, 1.14)),
                         sdSeg(q, vec2(0.085, 0.85), vec2(0.175, 1.14))) - 0.030;
        float legs = min(sdSeg(q, vec2(-0.09, 0.24), vec2(-0.11, 0.02)),
                         sdSeg(q, vec2(0.09, 0.24), vec2(0.11, 0.02))) - 0.050;
        float club = sdSeg(q, vec2(0.16, 0.40), vec2(0.46, 1.06)) - 0.058;
        return min(min(torso, head), min(min(horn, legs), club));
    }
    if (kind == 3) {
        // Tengu: long nose, high collar, one wing up.
        float robe = max(sdBox(q - vec2(0.0, 0.34), vec2(0.21, 0.30), 0.10),
                         -(q.y - 0.02));
        float head = length(q - vec2(0.0, 0.78)) - 0.135;
        float nose = sdSeg(q, vec2(0.05, 0.78), vec2(0.44, 0.70)) - 0.034;
        float wing = sdSeg(q, vec2(-0.14, 0.62), vec2(-0.44, 0.98)) - 0.040;
        float hat = sdBox(q - vec2(0.0, 0.93), vec2(0.10, 0.045), 0.020);
        return min(min(robe, head), min(min(nose, wing), hat));
    }
    if (kind == 4) {
        // Kappa: shell, beak, and the water dish on its head.
        float shell = length((q - vec2(0.0, 0.44)) * vec2(1.0, 1.22)) - 0.24;
        float head = length(q - vec2(0.02, 0.76)) - 0.135;
        float beak = sdSeg(q, vec2(0.10, 0.75), vec2(0.31, 0.71)) - 0.034;
        float dish = sdBox(q - vec2(0.02, 0.905), vec2(0.150, 0.026), 0.012);
        float legs = min(sdSeg(q, vec2(-0.09, 0.24), vec2(-0.13, 0.02)),
                         sdSeg(q, vec2(0.09, 0.24), vec2(0.13, 0.02))) - 0.042;
        return min(min(shell, head), min(min(beak, dish), legs));
    }
    // Hitotsume-kozo: the little one-eyed monk, small and eager.
    float robe = max(sdBox(q - vec2(0.0, 0.27), vec2(0.16, 0.25), 0.09),
                     -(q.y - 0.02));
    float head = length(q - vec2(0.0, 0.60)) - 0.145;
    float tuft = sdSeg(q, vec2(0.0, 0.72), vec2(0.02, 0.80)) - 0.022;
    return min(min(robe, head), tuft);
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float t = mod(iTime, 600.0);
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0);
    vec3 neon = palette(iHue);
    vec3 lampCol = mix(vec3(1.00, 0.66, 0.28), neon, 0.30);

    // ---- what the lanterns light -----------------------------------------
    // Summed once, used by every surface below. This is the only light in the
    // picture, which is why the street goes dark where the line has passed.
    float lit = 0.0;
    for (int i = 0; i < 7; i++) {
        vec4 L = iLamp[i];
        vec2 d = p - L.xy;
        d.y *= 1.25;
        lit += L.w / (1.0 + dot(d, d) * 26.0);
    }
    float glow = clamp(lit, 0.0, 3.0);

    // ---- night sky --------------------------------------------------------
    float up = clamp(v_uv.y, 0.0, 1.0);
    vec3 col = mix(vec3(0.035, 0.040, 0.085), vec3(0.010, 0.012, 0.038),
                   pow(up, 0.7));
    col = mix(col, palette(iHue + 0.55) * 0.20, 0.14);
    {
        // Moon, with a bite out of it for its phase, and a ring of cloud.
        vec2 mq = p - iMoon.xy;
        float m = length(mq) - 0.075;
        float bite = length(mq - vec2(iMoon.z * 0.085, 0.012)) - 0.072;
        float disc = max(m, -bite);
        col += vec3(0.95, 0.93, 0.82) * smoothstep(0.004, 0.0, disc) * 0.85;
        col += mix(vec3(0.8, 0.85, 1.0), neon, 0.3) * exp(-max(m, 0.0) * 9.0) * 0.30;
        float band = fbm(p * vec2(2.2, 5.0) + vec2(t * 0.03, 0.0));
        col += vec3(0.35, 0.38, 0.55) * smoothstep(0.52, 0.78, band)
             * smoothstep(0.02, 0.30, p.y) * 0.22;
    }

    // ---- the wall behind the street --------------------------------------
    // Shadows first: for a wall pixel, the point that would occlude it from a
    // given lamp is found by running the ray back from the lamp, and then
    // tested against the very silhouette the yokai is drawn from.
    float roofY = 0.300;
    if (p.y < roofY && p.y > ROAD - 0.02) {
        vec3 plaster = mix(vec3(0.090, 0.085, 0.100), palette(iHue + 0.42) * 0.10, 0.25);
        // Namako-kabe: raised plaster ribs over dark tile. Thin lines, or the
        // wall stops being a wall and becomes a chequerboard.
        vec2 g = vec2(p.x, p.y) * 21.0;
        float diamond = abs(fract(g.x + g.y) - 0.5) + abs(fract(g.x - g.y) - 0.5);
        plaster += vec3(0.10, 0.10, 0.11) * smoothstep(0.20, 0.05, diamond);
        float conduit = smoothstep(0.005, 0.0, abs(p.y - 0.252) - 0.0015);
        float shade = 0.0;
        for (int i = 0; i < 7; i++) {
            vec4 Y = iYokai[i];
            vec4 L = iLamp[i];
            // Backward ray: the occluder for this pixel, if there is one.
            vec2 back = L.xy + (p - L.xy) / 1.42;
            vec2 q = (back - vec2(Y.x, ROAD)) / Y.w;
            if (abs(q.x) > 0.75 || q.y < -0.1 || q.y > 1.25) continue;
            shade += smoothstep(0.012, -0.004, yokai(q, int(Y.y), Y.z));
        }
        shade = clamp(shade, 0.0, 1.0);
        col = plaster * (0.10 + glow * 0.80) * (1.0 - shade * 0.72);
        col += neon * conduit * (0.06 + iMid * 0.22) * (0.5 + iNeon * 0.6) * glow;
        // Tiled roof capping the wall.
        if (p.y > roofY - 0.055) {
            float tile = abs(fract(p.x * 26.0) - 0.5);
            col = mix(col, vec3(0.055, 0.050, 0.075) * (1.0 + tile * 0.8),
                      smoothstep(0.004, 0.0, p.y - roofY + 0.055));
        }
    }

    // ---- the road ---------------------------------------------------------
    if (p.y <= ROAD - 0.02) {
        vec3 earth = mix(vec3(0.10, 0.085, 0.075), palette(iHue + 0.1) * 0.20, 0.30);
        earth *= 0.80 + 0.40 * vnoise(p * 42.0 + iMarch);
        col = earth * (0.045 + glow * 0.85);
        // Wet sheen: each lamp gets a stretched reflection down the road.
        for (int i = 0; i < 7; i++) {
            vec4 L = iLamp[i];
            vec2 d = vec2(p.x - L.x, (p.y - (ROAD * 2.0 - L.y)) * 0.42);
            col += lampCol * L.w * 0.22 / (1.0 + dot(d, d) * 130.0);
        }
    }

    // ---- mist, which the lanterns light from inside -----------------------
    {
        float band = smoothstep(-0.50, -0.40, p.y) * smoothstep(-0.16, -0.32, p.y);
        float m = fbm(p * vec2(3.4, 7.0) + vec2(-iMarch * 0.4, t * 0.05));
        col += mix(vec3(0.5, 0.55, 0.68), lampCol, 0.55)
             * band * smoothstep(0.46, 0.80, m) * (0.03 + glow * 0.26) * iMist;
    }

    // ---- the parade -------------------------------------------------------
    for (int i = 0; i < 7; i++) {
        vec4 Y = iYokai[i];
        vec2 q = (p - vec2(Y.x, ROAD)) / Y.w;
        if (abs(q.x) > 0.80 || q.y < -0.12 || q.y > 1.30) continue;
        float d = yokai(q, int(Y.y), Y.z) * Y.w;
        if (d > 0.055) continue;
        // Key-plate black, read off a rim that is brightest on the side its
        // own lantern is on.
        vec4 L = iLamp[i];
        float side = clamp(dot(normalize(vec2(L.x, L.y) - vec2(Y.x, ROAD + Y.w * 0.5)),
                               normalize(p - vec2(Y.x, ROAD + Y.w * 0.5))), -1.0, 1.0);
        col = mix(col, vec3(0.012, 0.012, 0.030), smoothstep(0.005, 0.0, d));
        col += mix(lampCol, vec3(1.0), 0.25)
             * smoothstep(0.008, 0.0, abs(d))
             * (0.45 + 0.75 * max(side, 0.0)) * (0.6 + iNeon * 0.7) * (0.5 + L.w);
        // One eye, wherever this kind keeps it.
        int kind = int(Y.y);
        vec2 eye = kind == 0 ? vec2(0.0, 0.62)
                 : kind == 1 ? vec2(0.0, 0.60)
                 : kind == 3 ? vec2(-0.03, 0.80)
                 : kind == 5 ? vec2(0.0, 0.60) : vec2(0.0, 0.76);
        float e = length(q - eye) - (kind == 5 ? 0.075 : kind == 0 ? 0.056 : 0.030);
        col += vec3(1.0, 0.88, 0.45) * smoothstep(0.010, 0.0, e * Y.w)
             * (0.9 + iBeat * 0.8);
    }

    // ---- the lanterns themselves -----------------------------------------
    for (int i = 0; i < 7; i++) {
        vec4 L = iLamp[i];
        vec4 Y = iYokai[i];
        vec2 d = p - L.xy;
        if (abs(d.x) > 0.16 || abs(d.y) > 0.22) continue;
        // The pole, and the cord the lantern hangs on, both leaning with it.
        vec2 hand = vec2(Y.x + Y.w * 0.30, ROAD + Y.w * 0.50);
        col = mix(col, vec3(0.22, 0.15, 0.08),
                  smoothstep(0.003, 0.0, sdSeg(p, hand, L.xy + vec2(0.0, 0.062)) - 0.0038));
        // Chochin: ribbed paper drum, tilted by its own swing.
        float c = cos(L.z), s = sin(L.z);
        vec2 q = mat2(c, -s, s, c) * d;
        float body = sdBox(q, vec2(0.036, 0.050), 0.026);
        col = mix(col, mix(vec3(1.0, 0.80, 0.45), neon, 0.18) * (0.5 + L.w * 0.9),
                  smoothstep(0.004, 0.0, body));
        float rib = smoothstep(0.0035, 0.0, abs(fract(q.y * 22.0) - 0.5) - 0.44);
        col = mix(col, vec3(0.35, 0.18, 0.06), rib * smoothstep(0.002, 0.0, body) * 0.7);
        col = mix(col, vec3(0.16, 0.11, 0.05),
                  smoothstep(0.003, 0.0, abs(abs(q.y) - 0.050) - 0.008)
                  * step(abs(q.x), 0.030));
        // The filament, and the light coming off the paper.
        col += lampCol * exp(-length(d) * 20.0) * L.w * (0.7 + iBass * 0.45);
        col += vec3(1.0, 0.92, 0.70) * smoothstep(0.012, 0.0, length(q)) * L.w * 1.3;
        // Embers lifting off it.
        vec2 cell = vec2(d.x * 30.0, d.y * 30.0 - t * 2.2);
        vec2 cf = fract(cell) - 0.5;
        col += vec3(1.0, 0.65, 0.25) * step(0.955, hash21(floor(cell) + float(i)))
             * smoothstep(0.34, 0.02, length(cf))
             * step(d.y, 0.0) * L.w * (0.4 + iTreble * 0.9) * iGlitter;
    }

    // ---- seal --------------------------------------------------------------
    {
        vec2 q = p - vec2(A * 0.5 - 0.085, 0.40);
        float box = sdBox(q, vec2(0.030, 0.040), 0.006);
        col = mix(col, vec3(0.72, 0.10, 0.12), smoothstep(0.004, 0.0, box) * 0.9);
        col += mix(vec3(1.0, 0.35, 0.30), neon, 0.3)
             * smoothstep(0.004, 0.0, abs(box) - 0.004) * (0.6 + iBeat * 0.9);
        col = mix(col, vec3(0.98, 0.94, 0.88),
                  smoothstep(0.004, 0.0, abs(q.y) - 0.001) * step(abs(q.x), 0.018) * 0.8);
    }

    col *= 1.0 - 0.30 * smoothstep(0.60, 1.20, length(p));
    col = col / (1.0 + col * 0.28);
    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.90)), 1.0);
}
"""


class NightParade(BaseEffect):
    """Hyakki Yagyo: a procession that carries its own lighting with it."""

    NAME = 'Night Parade'
    AUTHOR = 'unicorn-viz'
    TAGS = ['ukiyo-e', 'neon', 'steampunk', 'groovy', 'energetic']
    PING_PONG_FRIENDS = ["Raijin's Drums", 'Floating World', 'Onden Watermill']

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.3)),
            'glitter': float(self.config.get('glitter', 1.0)),
            'neon_gain': float(self.config.get('neon_gain', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._glitter = max(0.0, float(self.parameters['glitter']))
        self._neon = max(0.0, float(self.parameters['neon_gain']))
        self._dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self._march = float(self.rng.uniform(0.0, _SPAN))
        self._moon = (float(self.rng.uniform(-0.6, 0.6)),
                      float(self.rng.uniform(0.37, 0.46)),
                      float(self.rng.uniform(-1.0, 1.0)))

        # The line: a shuffled cast so two activations are not the same parade.
        kinds = list(range(6)) + [int(self.rng.integers(0, 6)) for _ in range(2)]
        self.rng.shuffle(kinds)
        self._kind = np.array(kinds[:_NYOKAI], dtype=np.float64)
        # Big enough that a nose, a horn or a beak survives being drawn: at
        # 0.26-0.40 of the frame height none of the six kinds were telling
        # themselves apart and the line read as a row of bowling pins.
        # Tall enough for a nose or a horn to survive, short enough that the
        # tallest still stands under the roofline and several are in frame at
        # once -- it is a procession, so seeing three of them is not one.
        self._scale = np.array(
            [float(self.rng.uniform(0.33, 0.47)) for _ in range(_NYOKAI)])
        # Small ones walk with a quicker step than tall ones.
        self._stride = 6.5 / np.sqrt(self._scale / 0.40)
        self._phase = self.rng.uniform(0.0, _TAU, size=_NYOKAI)
        self._hop = np.zeros(_NYOKAI)

        # Each lantern is a damped pendulum on its pole.
        self._theta = self.rng.uniform(-0.12, 0.12, size=_NYOKAI)
        self._omega = np.zeros(_NYOKAI)
        self._prev_ax = np.zeros(_NYOKAI)
        self._burn = np.full(_NYOKAI, 0.7)

        self._tbl_y = np.zeros((_NYOKAI, 4), dtype='f4')
        self._tbl_l = np.zeros((_NYOKAI, 4), dtype='f4')

    def _x_of(self, i: int) -> float:
        """Where yokai *i* is standing, wrapped around the street."""
        x = (self._march + i * (_SPAN / _NYOKAI)) % _SPAN - _SPAN * 0.5
        return float(x)

    def update(self, dt: float, audio: AudioData) -> None:
        """Walk the line; the pole, the lantern and the light follow from it."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 1.8))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 1.8))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 1.8))
        beat_now = audio.beat > 0.5
        if beat_now:
            self._beat = 1.0
            # The whole line hops together, which is what sets every pendulum
            # going at once -- the swing is never scheduled, only provoked.
            self._hop[:] = 1.0
            self._omega += self.rng.uniform(-2.4, 2.4, size=_NYOKAI)
        self._beat = max(0.0, self._beat - dt * 3.2)
        self._hop = np.maximum(0.0, self._hop - dt * 2.6)

        pace = max(0.05, float(self.parameters['speed']))
        step = 0.16 * pace * (0.55 + self._bass * 0.85)
        self._march += self._dir * step * dt

        # The carrier's sideways acceleration drives the pendulum. Taking it
        # from the walk (rather than from a timer) is what makes the lantern
        # lag the step instead of matching it.
        bob = np.sin(self._march * self._stride + self._phase)
        ax = -np.cos(self._march * self._stride + self._phase) * step * 5.0
        drive = (ax - self._prev_ax) / max(dt, 1e-3)
        self._prev_ax = ax
        self._omega += (-26.0 * self._theta - 1.9 * self._omega
                        - np.clip(drive, -40.0, 40.0) * 0.020) * dt
        self._theta += self._omega * dt
        self._theta = np.clip(self._theta, -0.6, 0.6)
        self._bob = bob

        target = 0.55 + self._bass * 0.55 + self._hop * 0.35
        self._burn += (target - self._burn) * min(1.0, dt * 6.0)

    def render(self) -> None:
        """Upload the line and its lamps; the shader lights everything off them."""
        for i in range(_NYOKAI):
            x = self._x_of(i)
            sc = float(self._scale[i])
            bob = float(self._bob[i]) + float(self._hop[i]) * 0.9
            self._tbl_y[i] = (x, float(self._kind[i]), bob, sc)
            # The lantern hangs from a pole held out in front, and swings.
            hx, hy = x + sc * 0.30, _ROAD_Y + sc * 0.50
            th = float(self._theta[i])
            arm = sc * 0.62
            lx = hx + math.sin(th) * arm * 0.35 + arm * 0.30
            ly = hy + arm * 0.55 - (1.0 - math.cos(th)) * arm * 0.20
            ly += bob * 0.012
            self._tbl_l[i] = (lx, ly, th, float(self._burn[i]))

        p = self._prog
        p['iResolution'].value = (float(self.width), float(self.height))
        p['iTime'].value = float(self.time)
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iHue'].value = self._hue
        p['iGlitter'].value = self._glitter
        p['iNeon'].value = self._neon
        p['iMist'].value = float(np.clip(0.55 + self._mid * 0.6, 0.0, 1.5))
        p['iMarch'].value = float(self._march % 64.0)
        p['iMoon'].value = self._moon
        p['iYokai'].write(self._tbl_y.tobytes())
        p['iLamp'].write(self._tbl_l.tobytes())
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
