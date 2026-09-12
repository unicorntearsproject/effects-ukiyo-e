"""Raijin's Drums — the thunder god's ring, wired to the tempo and lit up.

Raijin is drawn the same way in every print from Sotatsu onward: crouched
inside a ring of taiko drums, bachi raised, the storm coming off the drums
rather than off him.  That iconography is already a percussion instrument, so
this effect treats it as one.

Where :mod:`floating_world` is a landscape built from plates and
:mod:`onden_watermill` is a machine downstream of one shaft, this is **an
instrument being played**, and the thing being modelled is the *strike*.

Nothing here fires on a timer.  A beat does not make thunder; a beat chooses
the next drum and starts the arm moving toward it, and the strike happens when
the arm arrives.  The swing is sized from the tempo, so the bachi lifts slowly
through the bar and comes down fast into the next beat -- it lands *on* the
beat because it was thrown a beat early, which is how a drummer does it.

Everything after the strike is downstream of it, in one chain:

    arm arrives -> drum head ripples -> a bolt leaves that drum ->
    the bolt lights the cloud it is inside -> the flash lights the rain,
    the ring, and the god

So the clouds do not flicker on their own.  They are lit from within, by a
bolt, at a position, and the rain brightens because that same flash reached
it.  Pull the strike out and the whole picture goes dark, which is the point.

It is the 1830s vocabulary run through neon and brass: the drum hoops are
brass with rivets, the lacing is neon rope, the bolts are edge-lit, and
Raijin himself stays a silhouette because a silhouette is what a woodblock
key plate actually gives you.

Audio reactivity:
  - bass    -> storm weight, the god's crouch, ground flash
  - mid     -> cloud churn and the ring's slow rotation
  - treble  -> rain density and the sparks off a struck head
  - beat    -> chooses the next drum and throws the arm at it

The drums are bound to the spectrum low-to-high around the ring, so which
drum gets chosen follows where the energy actually is in the track.

Randomised per activation: palette, storm seed, ring phase, which drum the
arm starts over, and the rain's slant.

Config keys (under ``[effects.RaijinDrums]``):
    speed, reactivity, glitter, neon_gain
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect
from unicornviz.vj_api import VJApi

__version__ = '0.4.0'

_NDRUM = 9
_NBOLT = 4
_RING_R = 0.36          # ring radius, in screen half-heights
_DRUM_R = 0.083
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
uniform float iFlash;       // thunder flash, 0..1, decaying from a strike
uniform float iCrouch;      // how far the god is bearing down, 0..1
uniform float iRain;        // rain density
uniform float iSlant;       // rain slant, shared with the cloud drift
uniform float iSky;
uniform vec4  iDrum[9];     // x, y, strike age (1 = just hit), band level
uniform vec4  iBolt[4];     // origin xy, life 1..0, seed
uniform vec4  iBachi[2];    // tip xy, shoulder xy
in  vec2 v_uv;
out vec4 fragColor;

#define TAU 6.28318530

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
    for (int i = 0; i < 4; i++) { s += a * vnoise(p); p *= 2.03; a *= 0.5; }
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

// A bolt as a jagged polyline generated from its seed, so four live bolts cost
// four seeds rather than four vertex lists.
float boltDist(vec2 p, vec2 a, vec2 b, float seed, out float along) {
    float best = 1e9;
    along = 0.0;
    vec2 prev = a;
    for (int i = 1; i <= 6; i++) {
        float u = float(i) / 6.0;
        vec2 q = mix(a, b, u);
        if (i < 6) {
            // Jitter perpendicular to the run, tapering to nothing at the tip.
            vec2 dir = normalize(b - a);
            vec2 nrm = vec2(-dir.y, dir.x);
            float j = (hash21(vec2(seed * 31.7 + float(i), seed * 11.3)) - 0.5);
            q += nrm * j * 0.16 * (1.0 - u * 0.5);
        }
        float d = sdSeg(p, prev, q);
        if (d < best) { best = d; along = u; }
        prev = q;
    }
    return best;
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float t = mod(iTime, 600.0);
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0);
    vec3 neon = palette(iHue);
    vec3 brass = vec3(0.72, 0.52, 0.20);

    // ---- storm sky -------------------------------------------------------
    // One bokashi wash, then cloud mass on top of it. The clouds are lit from
    // inside by the bolts further down, never on their own.
    float up = clamp(v_uv.y, 0.0, 1.0);
    vec3 lo, hi;
    if (iSky < 0.34)      { lo = vec3(0.070, 0.062, 0.125); hi = vec3(0.012, 0.016, 0.055); }
    else if (iSky < 0.67) { lo = vec3(0.115, 0.055, 0.060); hi = vec3(0.018, 0.012, 0.042); }
    else                  { lo = vec3(0.040, 0.078, 0.105); hi = vec3(0.010, 0.014, 0.048); }
    vec3 col = mix(lo, hi, pow(up, 0.75));
    col = mix(col, palette(iHue + 0.55) * 0.22, 0.12);

    vec2 cp = p * 2.3 + vec2(iSlant * 0.55, -t * 0.035);
    float cloud = fbm(cp + fbm(cp * 0.7 + t * 0.02) * 0.9);
    float mass = smoothstep(0.33, 0.82, cloud + iMid * 0.10);
    // Cloud mass stays dark on its own account -- it is meant to be lit by
    // the bolts, so tinting it up here is exactly what flattens the flash.
    col = mix(col, mix(vec3(0.075, 0.075, 0.115), palette(iHue + 0.4) * 0.28, 0.30),
              mass * 0.85);

    // ---- rain ------------------------------------------------------------
    // Sheets, slanted by the same wind that drifts the cloud.
    for (int s = 0; s < 2; s++) {
        float sc = 34.0 + float(s) * 21.0;
        vec2 rp = vec2(p.x + p.y * iSlant, p.y - t * (0.9 + float(s) * 0.5));
        vec2 ri = floor(rp * vec2(sc, sc * 0.28));
        float h = hash21(ri + float(s) * 19.0);
        if (h > 0.86 - iRain * 0.12) {
            vec2 rf = fract(rp * vec2(sc, sc * 0.28)) - 0.5;
            float line = smoothstep(0.22, 0.0, abs(rf.x) * 3.2)
                       * smoothstep(0.52, 0.1, abs(rf.y));
            col += mix(vec3(0.55, 0.66, 0.85), neon, 0.35) * line
                 * (0.10 + iFlash * 0.75 + iTreble * 0.18);
        }
    }

    // ---- lightning -------------------------------------------------------
    // Each bolt leaves the drum that was struck and runs down out of frame.
    // It writes into the cloud as well as onto the sheet, which is what makes
    // the sky look lit rather than tinted.
    for (int i = 0; i < 4; i++) {
        vec4 B = iBolt[i];
        if (B.z <= 0.0) continue;
        vec2 a = B.xy;
        float seed = B.w;
        // Runs outward from the ring and down, so bolts fan rather than stack.
        vec2 dir = normalize(a - vec2(0.0, -0.02) + vec2(0.0, -0.55));
        vec2 b = a + dir * 1.35;
        float along;
        float d = boltDist(p, a, b, seed, along);
        float life = B.z;
        // The tip is still travelling for the first part of the life.
        float reach = clamp((1.0 - life) * 3.2, 0.0, 1.0);
        if (along > reach) continue;
        float core = smoothstep(0.006, 0.0, d);
        float glow = exp(-d * 26.0);
        vec3 bc = mix(vec3(1.0), mix(neon, vec3(0.75, 0.85, 1.0), 0.5), 0.55);
        col += bc * (core * 1.5 + glow * 0.75) * life * (0.8 + iNeon * 0.6);
        // The cloud around the bolt, lit from inside.
        col += mix(bc, vec3(1.0), 0.3) * exp(-d * 4.6) * mass * life * 0.38;
    }

    // ---- the ring of drums ----------------------------------------------
    for (int i = 0; i < 9; i++) {
        vec4 D = iDrum[i];
        vec2 q = p - D.xy;
        if (abs(q.x) > 0.083 * 1.35 || abs(q.y) > 0.083 * 1.35) continue;
        float r = length(q);
        float ang = atan(q.y, q.x);
        float hit = D.z;
        float lvl = D.w;

        // A byo-uchi daiko seen face on: lacquered body, brass hoop with its
        // row of tacks, and a cream head shaded across its face. Drawing the
        // hoop as a thin edge instead of a filled ring is what left these
        // reading as flat grey counters.
        vec3 lacquer = mix(vec3(0.20, 0.045, 0.055), palette(iHue + 0.72) * 0.30, 0.35);
        col = mix(col, lacquer, smoothstep(0.005, 0.0, r - 0.083));
        vec3 skin = mix(vec3(0.93, 0.88, 0.76), palette(iHue + 0.12), 0.10);
        float lit = 0.84 + 0.40 * smoothstep(0.075, -0.03, r + q.y * 0.6);
        col = mix(col, skin * (lit + lvl * 0.25 + hit * 0.45),
                  smoothstep(0.004, 0.0, r - 0.083 * 0.80));
        float hoopR = 0.083 * 0.885;
        col = mix(col, brass * (0.80 + 0.25 * sin(ang * 3.0)),
                  smoothstep(0.004, 0.0, abs(r - hoopR) - 0.0085));
        float aa = (fract(ang / TAU * 16.0) - 0.5) * (TAU / 16.0) * r;
        col = mix(col, brass * 1.75,
                  smoothstep(0.0025, 0.0, length(vec2(aa, r - hoopR)) - 0.0052));

        if (hit > 0.0) {
            // The head's own ripple, running out from where it was struck.
            float ring = abs(r - (1.0 - hit) * 0.083 * 0.80);
            col += mix(neon, vec3(1.0), 0.45)
                 * smoothstep(0.018, 0.0, ring) * hit * 0.75;
            float sp = hash21(floor(q * 90.0) + float(i));
            col += vec3(1.0, 0.85, 0.55) * step(0.972, sp)
                 * smoothstep(0.083, 0.0, r) * hit * (0.6 + iTreble) * iGlitter;
        }
        // Neon rope lacing around the shell, brightening with the band.
        col += neon * smoothstep(0.013, 0.0, abs(r - 0.083))
             * (0.22 + lvl * 0.75 + hit * 0.9) * (0.6 + iNeon * 0.7);
    }

    // ---- Raijin ----------------------------------------------------------
    // A key-plate silhouette: a woodblock's key block gives you shape, not
    // shading, so he stays black and is read entirely off his rim.
    {
        vec2 c = vec2(0.0, -0.050 - iCrouch * 0.018);
        vec2 q = p - c;
        // Shoulders down to a crouching waist, knees pushed out to the sides.
        float torso = sdBox(q - vec2(0.0, 0.005), vec2(0.088, 0.070), 0.045);
        float shoulder = sdBox(q - vec2(0.0, 0.062), vec2(0.100, 0.022), 0.022);
        float hips = length((q - vec2(0.0, -0.072)) * vec2(1.0, 1.45)) - 0.082;
        float knee = min(length((q - vec2(-0.132, -0.098)) * vec2(1.0, 1.25)),
                         length((q - vec2(0.132, -0.098)) * vec2(1.0, 1.25))) - 0.050;
        float shin = min(sdSeg(q, vec2(-0.132, -0.098), vec2(-0.152, -0.182)),
                         sdSeg(q, vec2(0.132, -0.098), vec2(0.152, -0.182))) - 0.036;
        float head = length((q - vec2(0.0, 0.113)) * vec2(0.92, 1.0)) - 0.046;
        // Mane: a fan of spikes behind the head, which is how the prints draw
        // it -- three spikes on top read as ears instead.
        float hair = 1e9;
        for (int k = 0; k < 7; k++) {
            float aH = 3.1416 * (0.10 + 0.13 * float(k));
            float len = 0.085 + 0.040 * sin(float(k) * 2.1) + iBass * 0.010;
            vec2 root = vec2(0.0, 0.108);
            hair = min(hair, sdSeg(q, root, root + vec2(cos(aH), sin(aH)) * len)
                             - (0.024 - 0.013 * float(k) / 6.0));
        }
        // Two short horns.
        float horn = min(sdSeg(q, vec2(-0.030, 0.142), vec2(-0.052, 0.200)),
                         sdSeg(q, vec2(0.030, 0.142), vec2(0.052, 0.200))) - 0.014;
        float god = min(min(min(torso, shoulder), min(hips, knee)),
                        min(min(shin, head), min(hair, horn)));
        // Arms, drawn to wherever the CPU put the bachi this frame.
        for (int k = 0; k < 2; k++) {
            vec4 Bc = iBachi[k];
            god = min(god, sdSeg(p, Bc.zw, mix(Bc.zw, Bc.xy, 0.58)) - 0.028);
            float stick = sdSeg(p, mix(Bc.zw, Bc.xy, 0.52), Bc.xy) - 0.0095;
            col = mix(col, vec3(0.58, 0.42, 0.18), smoothstep(0.004, 0.0, stick));
            col += brass * smoothstep(0.020, 0.0, length(p - Bc.xy)) * (0.45 + iFlash);
        }
        col = mix(col, vec3(0.015, 0.015, 0.035), smoothstep(0.005, 0.0, god));
        // Circuit veins in the mass, the pack's steampunk tell.
        float vein = smoothstep(0.030, 0.0,
                                abs(fract(q.y * 15.0 + sin(q.x * 14.0) * 0.4) - 0.5));
        col += neon * vein * smoothstep(0.0, -0.020, god) * (0.20 + iFlash * 0.7);
        // Rim: it has to be narrower than the thinnest limb, or it stops
        // being a rim and floods the figure. At 0.017 against 0.021-thick
        // arms and 0.016-thick hair, every part of him glowed solid.
        col += mix(neon, vec3(1.0), 0.45)
             * smoothstep(0.007, 0.0, abs(god)) * (0.75 + iFlash * 1.5 + iBass * 0.45)
             * (0.7 + iNeon * 0.6);
        // A sash across the hips, and two eyes to look out of.
        col += mix(brass, neon, 0.4)
             * smoothstep(0.005, 0.0, abs(sdSeg(q, vec2(-0.085, -0.030),
                                                vec2(0.085, -0.052))) - 0.006)
             * smoothstep(0.0, -0.010, god) * (0.7 + iBeat * 0.6);
        float eye = min(length(q - vec2(-0.019, 0.120)),
                        length(q - vec2(0.019, 0.120))) - 0.0075;
        col += vec3(1.0, 0.90, 0.55) * smoothstep(0.005, 0.0, eye) * (1.3 + iFlash);
    }

    // ---- the flash on the sheet -----------------------------------------
    col += mix(neon, vec3(1.0), 0.55) * iFlash * 0.16;
    col *= 1.0 - 0.35 * smoothstep(0.55, 1.15, length(p));

    // ---- seal ------------------------------------------------------------
    {
        vec2 q = p - vec2(A * 0.5 - 0.085, -0.40);
        float box = sdBox(q, vec2(0.030, 0.040), 0.006);
        col = mix(col, vec3(0.72, 0.10, 0.12), smoothstep(0.004, 0.0, box) * 0.9);
        float gl = smoothstep(0.004, 0.0, abs(box) - 0.004);
        col += mix(vec3(1.0, 0.35, 0.30), neon, 0.3) * gl * (0.6 + iBeat * 0.9);
        col = mix(col, vec3(0.98, 0.94, 0.88),
                  smoothstep(0.004, 0.0, abs(q.y) - 0.001) * step(abs(q.x), 0.018) * 0.8);
    }

    col = col / (1.0 + col * 0.30);
    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.90)), 1.0);
}
"""


class RaijinDrums(BaseEffect):
    """Raijin's ring of taiko, where the arm arriving is what makes thunder."""

    NAME = "Raijin's Drums"
    AUTHOR = 'unicorn-viz'
    TAGS = ['ukiyo-e', 'neon', 'steampunk', 'intense', 'energetic', 'hard']
    PING_PONG_FRIENDS = ['Floating World', 'Onden Watermill', 'Audio Spectrum']

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
        self._flash = 0.0
        self._crouch = 0.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._sky = float(self.rng.uniform(0.0, 1.0))
        self._slant = float(self.rng.uniform(-0.45, 0.45))
        self._glitter = max(0.0, float(self.parameters['glitter']))
        self._neon = max(0.0, float(self.parameters['neon_gain']))
        self._ring_base = float(self.rng.uniform(-0.12, 0.12))
        self._ring_sway = float(self.rng.uniform(0.0, _TAU))
        self._ring_phase = self._ring_base

        self._level = np.zeros(_NDRUM, dtype=np.float64)
        self._age = np.zeros(_NDRUM, dtype=np.float64)
        self._from = int(self.rng.integers(0, _NDRUM))
        self._to = self._from
        self._swing = 1.0           # 1.0 = arrived
        self._swing_len = 0.5
        self._hand = 0              # which bachi is travelling
        self._bpm = 120.0
        # The swing is sized from the tempo actually observed, not from a
        # guess. Sizing it from a default BPM means that whenever the track is
        # faster than the guess, every beat re-throws the arm before it can
        # land -- and a strike only happens on arrival, so nothing ever fires.
        # Measured at three tempos with a guessed 0.5 s swing: 70, 60 and 90
        # beats, and zero strikes at all three.
        self._beat_iv = 0.5
        self._last_beat = -1.0
        self._vj_poll = 0.0

        self._bolts = np.zeros((_NBOLT, 4), dtype=np.float64)
        self._next_bolt = 0

        self._tbl_drum = np.zeros((_NDRUM, 4), dtype='f4')
        self._tbl_bolt = np.zeros((_NBOLT, 4), dtype='f4')
        self._tbl_bachi = np.zeros((2, 4), dtype='f4')

    # ------------------------------------------------------------ the ring
    def _drum_pos(self, i: int) -> tuple[float, float]:
        """Where drum *i* sits on the ring this frame.

        The ring turns slowly, so the low drums are not always in the same
        corner of the sheet.
        """
        # An arch over him, not a full ring. A 200-degree spread plus a
        # continuous rotation walks drums round underneath the figure, which
        # is neither how the prints stage it nor legible once he is in front
        # of them; 160 degrees with a slow sway keeps them all overhead.
        a = math.radians(170.0 - i * 20.0) + self._ring_phase
        return (math.cos(a) * _RING_R, -0.02 + math.sin(a) * _RING_R)

    def _poll_vj(self, dt: float) -> None:
        """Take the tempo from the VJ system when there is one.

        Polled and wrapped: the API is optional by contract and absent under
        tests and offscreen renders, so nothing here may be load-bearing.
        """
        self._vj_poll -= dt
        if self._vj_poll > 0.0:
            return
        self._vj_poll = 0.5
        api = VJApi.current()
        if api is None:
            return
        try:
            bpm = api.get_bpm()
        except Exception:            # noqa: BLE001 - context is never required
            return
        if bpm and 40.0 < float(bpm) < 220.0:
            self._bpm = float(bpm)
            # Only seed the interval; the measured value still governs, since
            # it is the one that actually matches the beats arriving here.
            if self._last_beat < 0.0:
                self._beat_iv = 60.0 / self._bpm

    # ------------------------------------------------------------------ sim
    def update(self, dt: float, audio: AudioData) -> None:
        """Choose a drum on the beat, throw the arm, and let it arrive."""
        super().update(dt, audio)
        self._poll_vj(dt)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 1.8))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 1.8))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 1.8))
        beat_now = audio.beat > 0.5
        if beat_now:
            self._beat = 1.0
        self._beat = max(0.0, self._beat - dt * 3.4)
        self._flash = max(0.0, self._flash - dt * 2.3)
        self._crouch += (self._bass - self._crouch) * min(1.0, dt * 5.0)
        self._ring_sway += dt * 0.22 * (0.4 + self._mid)
        self._ring_phase = self._ring_base + 0.085 * math.sin(self._ring_sway)

        # Bind the ring to the spectrum, low at one end and high at the other,
        # so which drum gets chosen follows where the track's energy is.
        bands = getattr(audio, 'bands', None)
        if bands is not None and len(bands) >= _NDRUM:
            edges = np.linspace(0, len(bands), _NDRUM + 1).astype(int)
            for i in range(_NDRUM):
                seg = bands[edges[i]:max(edges[i] + 1, edges[i + 1])]
                self._level[i] += (float(np.mean(seg)) - self._level[i]) * min(
                    1.0, dt * 9.0)
        else:
            self._level *= math.exp(-dt * 2.0)

        self._age = np.maximum(0.0, self._age - dt * 2.6)

        pace = max(0.05, float(self.parameters['speed']))
        if beat_now:
            # A beat does not make thunder. It picks the next drum and throws
            # the arm, sized so the stick arrives on the *following* beat --
            # which is how a drummer plays: the lift happens through the bar
            # and the strike is the arrival.
            if self._swing < 1.0:
                # The previous throw had not landed yet. Land it now rather
                # than re-throwing: the strike belongs to the beat, so a swing
                # sized from a tempo we read slightly wrong must never be able
                # to swallow a hit.
                self._strike(self._to)
                self._swing = 1.0
            if self._last_beat >= 0.0:
                iv = self.time - self._last_beat
                if 0.15 < iv < 2.2:
                    self._beat_iv += (iv - self._beat_iv) * 0.45
            self._last_beat = self.time
            self._from = self._to
            self._to = self._pick_drum()
            self._hand ^= 1
            self._swing = 0.0
            # Land a hair early, so the arrival reads as ahead of the beat
            # rather than smeared across it.
            self._swing_len = max(0.08, self._beat_iv * 0.94 / pace)

        if self._swing < 1.0:
            self._swing = min(1.0, self._swing + dt / self._swing_len)
            if self._swing >= 1.0:
                self._strike(self._to)

        self._bolts[:, 2] = np.maximum(0.0, self._bolts[:, 2] - dt * 1.15)

    def _pick_drum(self) -> int:
        """The loudest band that is not the drum just played."""
        w = self._level.copy()
        w[self._to] *= 0.25          # avoid hammering one drum all track
        if float(w.max()) <= 1e-6:
            return int(self.rng.integers(0, _NDRUM))
        return int(np.argmax(w))

    def _strike(self, i: int) -> None:
        """The arm arrived: ripple the head, fire a bolt, light the sky."""
        self._age[i] = 1.0
        self._flash = min(1.0, 0.55 + self._bass * 0.6)
        x, y = self._drum_pos(i)
        self._bolts[self._next_bolt] = (x, y, 1.0,
                                        float(self.rng.uniform(0.0, 90.0)))
        self._next_bolt = (self._next_bolt + 1) % _NBOLT

    def render(self) -> None:
        """Upload the ring, the live bolts and where the sticks are."""
        for i in range(_NDRUM):
            x, y = self._drum_pos(i)
            self._tbl_drum[i] = (x, y, self._age[i],
                                 float(np.clip(self._level[i] * 1.6, 0.0, 1.0)))
        self._tbl_bolt[:] = self._bolts.astype('f4')

        # The travelling stick eases up and then drops: slow through the lift,
        # fast into the arrival, so the strike reads as a strike.
        u = self._swing
        drop = u * u * u
        fx, fy = self._drum_pos(self._from)
        tx, ty = self._drum_pos(self._to)
        sx = fx + (tx - fx) * drop
        sy = fy + (ty - fy) * drop
        lift = math.sin(u * math.pi) * 0.12
        for k in range(2):
            shoulder = (-0.075 if k == 0 else 0.075, 0.035 - self._crouch * 0.02)
            if k == self._hand:
                tip = (sx, sy + lift)
            else:
                # The idle hand rests over the drum it last played.
                rx, ry = self._drum_pos(self._from)
                tip = (rx * 0.82, ry * 0.82 + 0.02)
            self._tbl_bachi[k] = (tip[0], tip[1], shoulder[0], shoulder[1])

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
        p['iFlash'].value = self._flash
        p['iCrouch'].value = float(np.clip(self._crouch, 0.0, 1.0))
        p['iRain'].value = float(np.clip(0.35 + self._treble * 0.7, 0.0, 1.4))
        p['iSlant'].value = self._slant
        p['iSky'].value = self._sky
        p['iDrum'].write(self._tbl_drum.tobytes())
        p['iBolt'].write(self._tbl_bolt.tobytes())
        p['iBachi'].write(self._tbl_bachi.tobytes())
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
