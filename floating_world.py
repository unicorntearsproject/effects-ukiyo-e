"""Floating World — neon ukiyo-e, printed one plate at a time and lit up.

A woodblock print is built from **plates**: one carved block per colour, each
inked and pressed onto the same sheet in order.  This effect works the same
way.  Every element -- the graded sky, the brass orrery, the cloud bands, Fuji,
the torii, the three wave plates, the drips, the rain, the blossoms, the
clockwork cranes, the seal -- is a plate with its own weight, and a director on
the CPU brings plates in and out over time.  So the picture is never finished:
it composes itself while you watch, which is the part a static print cannot do.

It is the 1830s vocabulary run through neon and brass rather than reproduced.
Hokusai's claw-foam still hooks off the crest, but it is edge-lit; Fuji still
sits in its bokashi gradient, but there are circuit veins in the slope and
something burning in the crater; the cranes are clockwork; the seal glows.

**The plates interact**, which is what stops it reading as parallax layers:

  - One wind value drives the cloud drift, the rain's slant, the blossom
    trajectories, the cranes' heading and the sway of every drip. They lean
    together because they are leaning on the same number.
  - The wave's crest is a position, not a decoration. Blossoms are torn off
    *at* the crest when the swell is running, drips are released from it, and
    the cranes lift over it as they pass.
  - The beat scatters the cranes, bursts blossoms off the crest, releases a
    fresh drip and flashes the seal.

**The director listens to the VJ system when there is one.**  ``VJApi.current()``
carries the song-structure hint auto-vj and the mixer publish -- a phrase role
of HOLD/RISE/PEAK/FALL/CLOSE -- and the printer works to it: a RISE inks plates
one after another, a PEAK has the whole picture on the sheet, a FALL wipes them
back, a CLOSE strips to the plates that carry the composition.  Tempo comes
from the same place and drives the orrery, so the gears turn on the beat rather
than near it.  None of it is required: with no app running -- tests, offscreen
renders, any standalone tool -- the director falls back to its own timer and
the effect behaves exactly as it does today.

Everything with a position -- blossoms, cranes, drips, gears -- is placed on
the CPU and handed to the shader ready to draw, so the fragment program is a
rasterizer rather than a simulation.

Audio reactivity:
  - bass    -> swell height, crest lift, gear speed, Fuji's core
  - mid     -> plate churn, cloud drift, sky gradient
  - treble  -> rain density, foam sparkle, blossom glow
  - beat    -> crane scatter, blossom burst, drip release, seal flash

Randomised per activation: neon hue, which plates start inked, wind direction,
crest position, the sky variant and ``self.time``.

Config keys (under ``[effects.FloatingWorld]``):
    speed, zoom, reactivity, glitter, neon_gain, layer_churn
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect
from unicornviz.vj_api import VJApi

__version__ = '0.1.0'

TAU = 2.0 * math.pi
_NPETAL = 16
_NCRANE = 6
_NDRIP = 10
_NGEAR = 4

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
uniform float iNeon;        // tweakable: how hot the neon runs
uniform float iWind;        // shared drift, wrapped so the noise cannot pop
uniform float iSwell;       // 0..1 how much sea is running
uniform float iCrest;       // x of the great crest, in world units
uniform float iCrestY;      // y the crest reaches, for the plates above it
uniform float iSky;         // per-activation sky variant

// Plate weights. The director fades these; a plate at 0 costs almost nothing
// because every layer is gated on its own weight.
uniform float iwGear;
uniform float iwCloud;
uniform float iwFuji;
uniform float iwTorii;
uniform float iwWave;
uniform float iwDrip;
uniform float iwRain;
uniform float iwPetal;
uniform float iwCrane;

// Placed on the CPU, ready to draw.
uniform vec4  iPetal[16];   // xy position, z rotation, w life
uniform vec4  iCrane[6];    // xy position, z flap phase, w scale
uniform vec4  iDrip[10];    // x, y top, z length, w life
uniform vec4  iGear[4];     // xy centre, z radius, w angle

in  vec2 v_uv;
out vec4 fragColor;

#define PI  3.14159265
#define TAU 6.28318530

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

// A carved edge: the key block's line, plus the glow the neon throws off it.
void ink(inout vec3 col, float d, float w, vec3 line, vec3 glow, float gain) {
    col = mix(col, line, smoothstep(w, w * 0.35, abs(d)));
    col += glow * exp(-abs(d) * 90.0) * gain;
}

// ── the sea ─────────────────────────────────────────────────────────────────
// Upper edge of one wave plate. The swell is three travelling harmonics; the
// great crest is a gaussian riding on top of them at iCrest.
float waveEdge(vec2 p, float base, float amp, float phase, float crestH) {
    float x = p.x;
    float y = base
            + amp * (0.055 * sin(x * 2.6 + phase)
                   + 0.032 * sin(x * 5.1 - phase * 1.37)
                   + 0.017 * sin(x * 9.7 + phase * 2.13));
    float d = x - iCrest;
    return y + crestH * exp(-d * d * 26.0);
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float tb = mod(iTime, 600.0);
    // A punch of field of view on the beat, the same trick the rides use.
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0) / max(iZoom * (1.0 - iPunch * 0.05), 0.05);

    vec3 neonA = palette(iHue);
    vec3 neonB = palette(iHue + 0.42);
    vec3 neonC = palette(iHue + 0.72);
    float gain = clamp(iNeon, 0.0, 3.0);

    // ── plate 1: the sheet ─────────────────────────────────────────────────
    // Foil washi. Long fibres one way, short the other, and an iridescent
    // sheen that only shows where the sheet catches the light.
    vec3 col = vec3(0.085, 0.075, 0.105);
    float fib = vnoise(vec2(p.x * 190.0, p.y * 26.0)) * 0.6
              + vnoise(p * 70.0) * 0.4;
    col *= 0.80 + fib * 0.45;
    // Foil sheen: an iridescent band that travels across the sheet, so the
    // paper catches the light rather than sitting there as a flat texture.
    float sheen = sin((p.x * 0.9 + p.y * 1.7) * 2.2 - tb * 0.22);
    col += palette(iHue + p.y * 0.35 + 0.5)
         * (0.05 + smoothstep(0.55, 1.0, sheen) * 0.11 * (0.5 + iMid * 0.9))
         * (0.4 + fib * 0.8);

    // ── plate 2: bokashi ───────────────────────────────────────────────────
    // The graded wash a printer pulls by wiping ink off the block, here in
    // neon. Two stops plus a horizon bloom.
    float up = clamp(p.y * 1.15 + 0.5, 0.0, 1.0);
    vec3 lo, hi;
    if (iSky < 0.34)      { lo = vec3(0.85, 0.16, 0.34); hi = vec3(0.05, 0.06, 0.26); }
    else if (iSky < 0.67) { lo = vec3(0.95, 0.45, 0.10); hi = vec3(0.04, 0.10, 0.30); }
    else                  { lo = vec3(0.20, 0.10, 0.55); hi = vec3(0.02, 0.14, 0.26); }
    vec3 sky = mix(lo, hi, pow(up, 0.85));
    sky = mix(sky, neonB, 0.18 + iMid * 0.14);
    // Bokashi is wiped by hand, so it bands. The steps are what makes the
    // wash read as printed rather than as a shader gradient.
    float steps = floor(up * 7.0) / 7.0;
    sky = mix(sky, sky * (0.86 + steps * 0.30), 0.55);
    sky += neonB * smoothstep(0.72, 1.0, sin(up * 42.0)) * 0.020;
    col = mix(col, sky, 0.86);
    // The disc: a reactor sun sitting on the gradient.
    vec2 sunP = vec2(-0.42 + sin(iWind * 0.05) * 0.05, 0.24);
    float sd = length(p - sunP);
    col += mix(vec3(1.0, 0.45, 0.22), neonA, 0.35)
         * (smoothstep(0.155, 0.148, sd) * 0.55 + exp(-sd * 7.0) * 0.30)
         * (0.9 + iBass * 0.6) * gain;
    ink(col, sd - 0.152, 0.004, vec3(1.0, 0.85, 0.6), neonA, 0.35 * gain);

    // ── plate 3: the orrery ────────────────────────────────────────────────
    if (iwGear > 0.01) {
        for (int i = 0; i < 4; i++) {
            vec4 G = iGear[i];
            vec2 q = p - G.xy;
            float r = length(q);
            if (r > G.z * 1.35) continue;
            float a = atan(q.y, q.x) + G.w;
            // Teeth: the rim steps out on alternate angular cells.
            float teeth = G.z * (1.0 + 0.085 * step(0.5, fract(a / TAU * 14.0)));
            float rim = smoothstep(0.016, 0.0, abs(r - teeth));
            float hub = smoothstep(0.013, 0.0, abs(r - G.z * 0.24));
            float spoke = smoothstep(0.010, 0.0, abs(sin(a * 3.0)) * r)
                        * step(r, G.z * 0.95);
            // A faint plate behind the spokes so the wheel has a body.
            float disc = smoothstep(0.004, 0.0, r - teeth) * 0.28;
            float m = clamp(rim + hub + spoke + disc, 0.0, 1.0) * iwGear;
            vec3 brass = mix(vec3(0.72, 0.50, 0.18), neonC, 0.12);
            col = mix(col, brass * (0.65 + iMid * 0.45), m);
            col += mix(vec3(1.0, 0.78, 0.35), neonC, 0.3) * m * 0.22 * gain
                 * (0.35 + iBass * 0.6);
        }
    }

    // ── plate 4: kumogata ──────────────────────────────────────────────────
    // Cloud bands: the flat stacked shapes of a printed sky, chrome-edged.
    if (iwCloud > 0.01) {
        for (int i = 0; i < 3; i++) {
            float fi = float(i);
            float by = 0.30 - fi * 0.085;
            float drift = iWind * (0.055 + fi * 0.014) + fi * 2.3;
            // Kumogata are cut as stacked lobes, not as a soft edge, so the
            // profile is deliberately bumpy and the band has a hard bottom.
            float lobes = 0.026 * sin(p.x * 9.0 + drift)
                        + 0.017 * sin(p.x * 15.5 - drift * 1.4)
                        + 0.011 * sin(p.x * 27.0 + drift * 0.7);
            float d = p.y - (by + lobes);
            float band = smoothstep(0.0, -0.004, d) * smoothstep(-0.040, -0.034, d);
            vec3 cc = mix(vec3(0.86, 0.88, 0.94), neonB, 0.25 + fi * 0.12);
            col = mix(col, cc * (0.5 + iMid * 0.35), band * iwCloud * 0.88);
            col += neonB * smoothstep(0.006, 0.0, abs(d)) * iwCloud * 0.5 * gain;
        }
    }

    // ── plate 5: Fuji ──────────────────────────────────────────────────────
    if (iwFuji > 0.01) {
        vec2 q = p - vec2(0.27, 0.015);
        float h = 0.34;
        float t = clamp(q.y / h, 0.0, 1.0);
        // The concave flare that makes the silhouette read as Fuji and not as
        // a triangle: the slope has to fall away faster near the summit than a
        // straight edge would.
        float edge = 0.40 * pow(1.0 - t, 1.85);
        float d = abs(q.x) - edge;
        float body = step(d, 0.0) * step(0.0, q.y) * step(q.y, h);
        if (body > 0.0) {
            // Indigo, the way the block would be inked, with only a touch of
            // the neon accent -- taking the palette straight gave the mountain
            // whatever hue the run happened to draw, including green.
            vec3 rock = mix(vec3(0.09, 0.10, 0.26), neonC, 0.07);
            // The sun is off to the left, so the left slope catches it.
            rock *= 1.0 + 0.55 * smoothstep(0.0, -0.30, q.x / max(edge, 1e-3));
            rock *= 0.75 + 0.35 * t;            // and the base sits in shadow
            // Circuit veins running down the slope.
            float vein = smoothstep(0.80, 1.0,
                                    vnoise(vec2(q.x * 70.0, q.y * 26.0 - 1.7)));
            rock += neonA * vein * (0.12 + iBass * 0.45 + iMid * 0.22) * gain;
            // Snow, with the ragged line a carver would cut.
            float snowline = h * 0.60 + 0.016 * sin(q.x * 44.0)
                                      + 0.012 * vnoise(vec2(q.x * 55.0, 3.0));
            float snow = smoothstep(snowline - 0.012, snowline + 0.012, q.y);
            rock = mix(rock, mix(vec3(0.93, 0.96, 1.0), neonB, 0.16), snow * 0.96);
            col = mix(col, rock, iwFuji);
            // Snow runs down the gullies below the line, the way a carver cuts
            // it -- a hard horizontal edge would read as a painted stripe.
            float gully = smoothstep(0.55, 0.95, vnoise(vec2(q.x * 26.0, 7.0)));
            rock = mix(rock, vec3(0.93, 0.96, 1.0),
                       gully * smoothstep(snowline - 0.10, snowline, q.y) * 0.8);
            col = mix(col, rock, iwFuji);
            // Something burning in the crater.
            float crater = smoothstep(0.055, 0.0, length(q - vec2(0.0, h)));
            col += mix(vec3(1.0, 0.35, 0.10), neonA, 0.3) * crater * iwFuji
                 * (0.5 + iBass * 1.5) * gain;
        }
        ink(col, max(d, max(-q.y, q.y - h)), 0.0035,
            vec3(0.02, 0.02, 0.05), neonA, 0.30 * iwFuji * gain);
    }

    // ── plate 6: torii ─────────────────────────────────────────────────────
    if (iwTorii > 0.01) {
        vec2 q = p - vec2(-0.30, -0.16);
        float post = min(sdBox(q - vec2(-0.105, 0.075), vec2(0.014, 0.115)),
                         sdBox(q - vec2( 0.105, 0.075), vec2(0.014, 0.115)));
        // The kasagi lifts at the ends, so it is drawn as a shallow arc.
        float lift = 0.020 * q.x * q.x * 46.0;
        float top = sdBox(q - vec2(0.0, 0.192 + lift), vec2(0.150, 0.011));
        float nuki = sdBox(q - vec2(0.0, 0.148), vec2(0.118, 0.008));
        float d = min(post, min(top, nuki));
        vec3 lac = mix(vec3(0.72, 0.10, 0.09), neonA, 0.30);
        col = mix(col, lac * (0.7 + iMid * 0.5), smoothstep(0.002, 0.0, d) * iwTorii);
        col += neonA * exp(-abs(d) * 70.0) * iwTorii * 0.45 * gain * (0.5 + iBass * 0.6);
    }

    // ── plate 7: the sea ───────────────────────────────────────────────────
    // Three plates, back to front, so the water reads as printed layers.
    float crestH = iCrestY;
    if (iwWave > 0.01) {
        // Far plate first. Drawing near-to-far instead lets the furthest
        // plate -- which sits highest on the sheet and fills everything below
        // its own edge -- paint straight over the two in front of it, which is
        // why the sea came out as one flat field.
        for (int i = 0; i < 3; i++) {
            float fi = float(i);
            float base = -0.09 - fi * 0.115;
            float amp = iSwell * (0.45 + fi * 0.38);
            float ph = iWind * (0.30 + fi * 0.10) + fi * 1.9;
            float ch = crestH * (1.05 - fi * 0.22);
            float ey = waveEdge(p, base, amp, ph, ch);
            float d = p.y - ey;
            if (d < 0.09) {
                // Nearer plates print darker, the way successive impressions
                // of the same block build up.
                vec3 deep = mix(vec3(0.07, 0.20, 0.50), vec3(0.02, 0.05, 0.17),
                                fi * 0.5);
                deep = mix(deep, neonC, 0.12 - fi * 0.03);
                col = mix(col, deep, step(d, 0.0) * iwWave);
                // Foam: a bright band riding the edge, thicker on the crest.
                float foam = smoothstep(0.026, 0.0, abs(d + 0.009))
                           * (0.55 + iTreble * 0.8);
                col = mix(col, mix(vec3(0.92, 0.96, 1.0), neonB, 0.22),
                          foam * iwWave * 0.85);
                // The claws Hokusai hooks off the crest: a row of fingers
                // curling over the top, biggest at the centre.
                // Four fingers climbing over the crest and reaching out past
                // it, each smaller than the last -- a curl, rather than the row
                // of equal bubbles a straight line of circles gives you.
                float lean = sign(iCrest) == 0.0 ? 1.0 : -sign(iCrest);
                for (int k = 0; k < 4; k++) {
                    float fk = float(k);
                    float cx = iCrest + lean * (0.012 + fk * 0.055) + fi * 0.02;
                    float cyy = waveEdge(vec2(cx, 0.0), base, amp, ph, ch);
                    float rr = (0.040 - fk * 0.0075) * (0.75 + iBass * 0.55)
                             * (1.0 - fi * 0.18);
                    vec2 cc = vec2(cx, cyy + rr * 0.5 + fk * 0.016);
                    vec2 cq = p - cc;
                    float cd = length(cq) - rr;
                    // Each finger drags a tapered tip out along the curl.
                    vec2 tq = cq - vec2(lean * rr * 1.15, rr * 0.55);
                    float tip = length(tq * vec2(1.0, 2.2)) - rr * 0.34;
                    cd = min(cd, tip);
                    col = mix(col, vec3(0.96, 0.98, 1.0),
                              smoothstep(0.005, 0.0, cd) * iwWave);
                    // Key-block line around each finger, or they merge into
                    // one white mass instead of reading as separate claws.
                    col = mix(col, vec3(0.05, 0.06, 0.16),
                              smoothstep(0.0035, 0.0, abs(cd)) * iwWave * 0.85);
                    col += neonB * exp(-max(cd, 0.0) * 42.0) * iwWave * 0.45 * gain;
                }
                ink(col, d, 0.0028, vec3(0.02, 0.03, 0.10), neonB,
                    0.40 * iwWave * gain);
            }
        }
    }

    // ── plate 8: the drips ─────────────────────────────────────────────────
    // Ink running off the plates. Released at the crest, so they only ever
    // appear where the sea has just been.
    if (iwDrip > 0.01) {
        for (int i = 0; i < 10; i++) {
            vec4 D = iDrip[i];
            if (D.w <= 0.0) continue;
            float dx = p.x - (D.x + sin(p.y * 9.0 + iWind * 0.7) * 0.006);
            if (abs(dx) > 0.02) continue;
            // Attached to the sea rather than to a remembered height: the
            // middle plate's edge is evaluated right here, at this drip's x.
            float top = waveEdge(vec2(D.x, 0.0), -0.205, iSwell * 0.83,
                                 iWind * 0.40 + 1.9, crestH * 0.83) - 0.004;
            float bot = top - D.z;
            if (p.y > top || p.y < bot - 0.02) continue;
            // The run tapers, and the bead at the end is fatter.
            float t = clamp((top - p.y) / max(D.z, 1e-3), 0.0, 1.0);
            float w = 0.0034 * (1.0 - t * 0.5);
            float d = abs(dx) - w;
            float bead = length(vec2(dx, p.y - bot)) - 0.0058;
            d = min(d, bead);
            vec3 dc = mix(neonA, neonB, t);
            col = mix(col, dc, smoothstep(0.0022, 0.0, d) * D.w * iwDrip);
            col += dc * exp(-max(d, 0.0) * 90.0) * D.w * iwDrip * 0.55 * gain;
        }
    }

    // ── plate 9: the shower ────────────────────────────────────────────────
    // Hiroshige's rain: straight parallel strokes, all leaning the same way as
    // everything else, because they share the wind.
    if (iwRain > 0.01) {
        float lean = 0.30 + sin(iWind * 0.03) * 0.22;
        vec2 rp = vec2(p.x + p.y * lean, p.y);
        float lane = floor(rp.x * 78.0);
        float lh = hash21(vec2(lane, 11.0));
        float run = fract(rp.y * 3.1 - iWind * (0.9 + lh * 0.7) - lh * 5.0);
        float stroke = smoothstep(0.20, 0.0, abs(fract(rp.x * 78.0) - 0.5) * 2.0)
                     * smoothstep(0.62, 0.80, run) * smoothstep(1.0, 0.90, run);
        col += mix(vec3(0.85, 0.92, 1.0), neonB, 0.30) * stroke
             * step(0.72, lh) * iwRain * (0.22 + iTreble * 0.55) * gain
             * (0.35 + iGlitter * 0.85);
    }

    // ── plate 10: blossom ──────────────────────────────────────────────────
    if (iwPetal > 0.01) {
        vec3 petA = mix(vec3(1.0, 0.72, 0.84), neonA, 0.30);
        vec3 petB = mix(vec3(1.0, 0.35, 0.55), neonB, 0.30);
        for (int i = 0; i < 16; i++) {
            vec4 P = iPetal[i];
            if (P.w <= 0.0) continue;
            vec2 q = p - P.xy;
            if (dot(q, q) > 0.0016) continue;
            q = rot(P.z) * q;
            // A petal is a teardrop with a notch cut in the wide end.
            float d = length(q * vec2(1.0, 1.9)) - 0.011;
            d = max(d, -(length((q - vec2(0.0, 0.011)) * vec2(1.0, 1.6)) - 0.006));
            vec3 pc = mix(petA, petB, fract(P.z * 0.6));
            col = mix(col, pc, smoothstep(0.0022, 0.0, d) * P.w);
            col += pc * exp(-max(d, 0.0) * 110.0) * P.w * 0.5 * gain
                 * (0.4 + iTreble * 0.9) * (0.4 + iGlitter * 0.8);
        }
    }

    // ── plate 11: the cranes ───────────────────────────────────────────────
    if (iwCrane > 0.01) {
        vec3 body = mix(vec3(0.95, 0.96, 1.0), neonC, 0.18);
        for (int i = 0; i < 6; i++) {
            vec4 C = iCrane[i];
            vec2 q = (p - C.xy) / max(C.w, 1e-3);
            if (dot(q, q) > 4.0) continue;
            float flap = sin(C.z) * 0.55;
            // Two swept wings meeting at the body, plus a neck and a bill.
            float wing = abs(q.y - abs(q.x) * flap) - 0.06;
            wing = max(wing, abs(q.x) - 1.0);
            float neck = sdBox(rot(-0.5) * (q - vec2(0.42, 0.10)), vec2(0.34, 0.035));
            float d = min(wing, neck);
            col = mix(col, body, smoothstep(0.035, 0.0, d) * iwCrane);
            col += neonC * exp(-max(d, 0.0) * 26.0) * iwCrane * 0.35 * gain;
            // Clockwork: a lit eye.
            col += vec3(1.0, 0.35, 0.25)
                 * smoothstep(0.05, 0.0, length(q - vec2(0.70, 0.20)))
                 * iwCrane * (0.6 + iBeat * 1.4) * gain;
        }
    }

    // ── plate 12: the seal ─────────────────────────────────────────────────
    // The hanko a printer stamps in the corner, here lit from behind.
    vec2 sq = p - vec2(A * 0.5 - 0.085, -0.5 + 0.085);
    float sbox = sdBox(sq, vec2(0.036, 0.036));
    float carve = min(min(sdBox(sq - vec2(0.0,  0.014), vec2(0.020, 0.004)),
                          sdBox(sq - vec2(0.0, -0.006), vec2(0.020, 0.004))),
                      sdBox(sq - vec2(0.0, -0.024), vec2(0.020, 0.004)));
    carve = min(carve, sdBox(sq - vec2(-0.020, 0.0), vec2(0.004, 0.030)));
    vec3 seal = mix(vec3(0.86, 0.12, 0.10), neonA, 0.25);
    // The stamp is the block minus the carved-away strokes.
    float sealM = smoothstep(0.003, 0.0, sbox) * smoothstep(0.0, 0.003, carve);
    col = mix(col, seal * (0.7 + iBeat * 0.9), sealM);
    col += seal * exp(-abs(sbox) * 60.0) * (0.25 + iBeat * 0.8) * gain;

    // The printed edge of the sheet.
    float border = sdBox(p, vec2(A * 0.5 - 0.020, 0.5 - 0.020));
    col = mix(col, vec3(0.05, 0.05, 0.08), smoothstep(0.0, 0.003, border));
    col += neonB * exp(-abs(border) * 120.0) * 0.30 * gain;

    col += neonA * iBeat * 0.06;
    col *= smoothstep(1.45, 0.30, length(v_uv - 0.5));
    col = col / (1.0 + col * 0.42);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.86));
    fragColor = vec4(col, 1.0);
}
"""


class FloatingWorld(BaseEffect):
    """Neon ukiyo-e composed live, one woodblock plate at a time."""

    NAME = 'Floating World'
    AUTHOR = 'unicorn-viz'
    TAGS = ['ukiyo-e', 'neon', 'steampunk', 'groovy', 'energetic', 'showcase']
    PING_PONG_FRIENDS = ['Audio Spectrum', 'Audio Waveforms']

    # Plates the director may ink. The sheet, the bokashi wash, the seal and
    # the border are always printed; these are the ones that come and go.
    _PLATES = ('gear', 'cloud', 'fuji', 'torii', 'wave', 'drip',
               'rain', 'petal', 'crane')
    # Plates that carry the picture: at least two of these stay inked, or the
    # sheet empties out and the effect has nothing on it.
    _ANCHORS = ('fuji', 'wave', 'cloud', 'torii')
    # How full the sheet should be for each phrase role the VJ system reports.
    # A rise builds the picture up, a peak has everything on it, a close strips
    # back to what carries the composition.
    _ROLE_PLATES = {'HOLD': 5, 'RISE': 7, 'PEAK': 9, 'FALL': 4, 'CLOSE': 3}

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'zoom': float(self.config.get('zoom', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.3)),
            'glitter': float(self.config.get('glitter', 1.0)),
            'neon_gain': float(self.config.get('neon_gain', 1.0)),
            'layer_churn': float(self.config.get('layer_churn', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._punch = 0.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._sky = float(self.rng.uniform(0.0, 1.0))
        self._wind = float(self.rng.uniform(0.0, 100.0))
        self._wind_dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self._swell = 0.5
        self._crest = float(self.rng.uniform(-0.35, 0.35))
        self._crest_v = float(self.rng.uniform(-0.05, 0.05))
        self._crest_y = 0.10

        # Plate weights, and where the director is taking them. A random
        # subset starts inked so no two activations open on the same picture.
        self._w = {k: 0.0 for k in self._PLATES}
        self._target = {k: 0.0 for k in self._PLATES}
        for k in self._PLATES:
            on = self.rng.random() < 0.55
            self._w[k] = 1.0 if on else 0.0
            self._target[k] = self._w[k]
        self._ensure_anchors()
        self._churn_cd = float(self.rng.uniform(2.0, 6.0))
        # Runtime context, when an app is running to provide it.
        self._role = ''
        self._bpm = 120.0
        self._vj_poll = 0.0

        self._petals = np.zeros((_NPETAL, 6), dtype=np.float64)   # x y vx vy rot life
        self._cranes = np.zeros((_NCRANE, 5), dtype=np.float64)   # x y flap scale vx
        self._drips = np.zeros((_NDRIP, 4), dtype=np.float64)     # x ytop len life
        self._gears = np.zeros((_NGEAR, 4), dtype=np.float64)     # x y r ang
        for i in range(_NGEAR):
            self._gears[i] = (float(self.rng.uniform(-0.8, 0.8)),
                              float(self.rng.uniform(0.05, 0.42)),
                              float(self.rng.uniform(0.045, 0.11)),
                              float(self.rng.uniform(0.0, TAU)))
        for i in range(_NCRANE):
            self._spawn_crane(i)

        self._tbl_petal = np.zeros((_NPETAL, 4), dtype='f4')
        self._tbl_crane = np.zeros((_NCRANE, 4), dtype='f4')
        self._tbl_drip = np.zeros((_NDRIP, 4), dtype='f4')
        self._tbl_gear = np.zeros((_NGEAR, 4), dtype='f4')

    # ----------------------------------------------------------- the director
    def _ensure_anchors(self) -> None:
        """Keep at least two carrying plates inked, so the sheet is never bare."""
        lit = [k for k in self._ANCHORS if self._target[k] > 0.5]
        while len(lit) < 2:
            k = self._ANCHORS[int(self.rng.integers(len(self._ANCHORS)))]
            if k not in lit:
                self._target[k] = 1.0
                lit.append(k)

    def _read_runtime(self, dt: float) -> None:
        """Pick up phrase role and tempo from the VJ API, if there is one.

        Polled rather than read per frame, and wrapped: the API is optional by
        contract and absent under tests, offscreen renders and standalone
        tools, so nothing here may be load-bearing.
        """
        self._vj_poll -= dt
        if self._vj_poll > 0.0:
            return
        self._vj_poll = 0.25
        api = VJApi.current()
        if api is None:
            self._role = ''
            return
        try:
            section = api.get_section()
            bpm = api.get_bpm()
        except Exception:            # noqa: BLE001 - context is never required
            self._role = ''
            return
        role = str(section.get('role', '')).upper() if section else ''
        if role != self._role:
            # A new phrase is a new picture: re-cut the plates now rather than
            # waiting out the timer.
            self._role = role
            self._churn_cd = 0.0
        if bpm and bpm > 20.0:
            self._bpm = float(bpm)

    def _work_toward_role(self) -> None:
        """Ink or wipe one plate, moving the sheet toward the phrase's fullness."""
        want = self._ROLE_PLATES.get(self._role, 0)
        if not want:
            self._churn()
            return
        lit = [k for k in self._PLATES if self._target[k] > 0.5]
        dark = [k for k in self._PLATES if self._target[k] <= 0.5]
        if len(lit) < want and dark:
            self._target[dark[int(self.rng.integers(len(dark)))]] = 1.0
        elif len(lit) > want and lit:
            self._target[lit[int(self.rng.integers(len(lit)))]] = 0.0
        else:
            self._churn()
        self._ensure_anchors()

    def _churn(self) -> None:
        """Ink one plate or wipe another, the way a printer works up a picture."""
        k = self._PLATES[int(self.rng.integers(len(self._PLATES)))]
        self._target[k] = 0.0 if self._target[k] > 0.5 else 1.0
        self._ensure_anchors()

    # ------------------------------------------------------------- the actors
    def _spawn_crane(self, i: int) -> None:
        side = 1.0 if self.rng.random() < 0.5 else -1.0
        self._cranes[i] = (
            -side * float(self.rng.uniform(0.9, 1.5)),
            float(self.rng.uniform(0.02, 0.40)),
            float(self.rng.uniform(0.0, TAU)),
            float(self.rng.uniform(0.055, 0.100)),
            side * float(self.rng.uniform(0.10, 0.22)),
        )

    def _spawn_petal(self, i: int, at_crest: bool) -> None:
        if at_crest:
            # Torn off the crest itself, and thrown along the wind.
            x = self._crest + float(self.rng.uniform(-0.10, 0.10))
            y = -0.16 + self._crest_y * 0.9 + float(self.rng.uniform(0.0, 0.06))
            vx = self._wind_dir * float(self.rng.uniform(0.12, 0.34))
            vy = float(self.rng.uniform(0.10, 0.30))
        else:
            x = -self._wind_dir * float(self.rng.uniform(0.8, 1.2))
            y = float(self.rng.uniform(-0.1, 0.45))
            vx = self._wind_dir * float(self.rng.uniform(0.08, 0.20))
            vy = float(self.rng.uniform(-0.06, 0.02))
        self._petals[i] = (x, y, vx, vy, float(self.rng.uniform(0.0, TAU)), 1.0)

    def _spawn_drip(self, i: int) -> None:
        """Release a run of ink from the foam, somewhere near the crest.

        Only the x is chosen here. The shader hangs the run off the wave's own
        edge at that x, so a drip is always attached to water however the swell
        happens to be standing -- no second copy of the wave model to drift.
        """
        self._drips[i] = (self._crest + float(self.rng.uniform(-0.20, 0.20)),
                          0.0, 0.0, 1.0)

    def update(self, dt: float, audio: AudioData) -> None:
        """Run the director, the wind, and everything the wind carries."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 2.0))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 2.0))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 2.0))
        beat = audio.beat > 0.5
        if beat:
            self._beat = 1.0
            self._punch = 1.0
        self._beat = max(0.0, self._beat - dt * 4.0)
        self._punch = max(0.0, self._punch - dt * 2.4)

        speed = max(0.05, float(self.parameters['speed']))
        step = dt * speed

        # One wind, shared by every plate that moves. Wrapped at 100 so the
        # noise and the drifts cannot pop when it comes round.
        self._wind = (self._wind
                      + step * self._wind_dir * (0.55 + self._mid * 1.1)) % 100.0

        # The sea. The swell follows the low end but lags it, so the water has
        # weight; the crest wanders and turns at the edges of the sheet.
        swell_t = 0.35 + min(1.0, self._bass) * 0.75
        self._swell += (swell_t - self._swell) * min(1.0, dt * 2.2)
        self._crest_y = 0.05 + self._swell * 0.20 + self._bass * 0.06
        self._crest += step * self._crest_v * 6.0
        if abs(self._crest) > 0.42:
            self._crest = math.copysign(0.42, self._crest)
            self._crest_v = -self._crest_v

        # The director. Plates cross-fade rather than cutting.
        self._read_runtime(dt)
        churn = max(0.0, float(self.parameters['layer_churn']))
        self._churn_cd -= step * (0.6 + self._mid * 0.8) * churn
        if self._churn_cd <= 0.0 and churn > 0.0:
            self._work_toward_role()
            # A rise or a fall is going somewhere, so it re-cuts sooner.
            quick = self._role in ('RISE', 'FALL')
            self._churn_cd = float(self.rng.uniform(1.2, 3.0) if quick
                                   else self.rng.uniform(3.0, 9.0))
        for k in self._PLATES:
            self._w[k] += (self._target[k] - self._w[k]) * min(1.0, dt * 1.3)

        # Gears turn with the low end; alternate ones counter-rotate, because a
        # train of gears that all turned the same way would be a lie.
        for i in range(_NGEAR):
            sign = 1.0 if i % 2 == 0 else -1.0
            self._gears[i, 3] += (step * sign * (0.5 + self._bass * 2.6)
                                  * (self._bpm / 120.0))

        # Cranes ride the wind, and lift as they pass over the crest -- the
        # plates are meant to notice each other.
        for i in range(_NCRANE):
            c = self._cranes[i]
            c[0] += step * c[4] * (1.0 + self._mid * 0.5)
            c[2] += dt * (7.0 + self._bass * 9.0)
            over = math.exp(-((c[0] - self._crest) ** 2) * 22.0)
            c[1] += step * over * self._crest_y * 1.6
            if beat:
                c[1] += float(self.rng.uniform(-0.03, 0.05))
            c[1] = min(0.46, max(-0.05, c[1]))
            if abs(c[0]) > 1.5:
                self._spawn_crane(i)

        # Blossom. Petals are torn off the crest while the sea is running, and
        # blow in from the windward edge when it is not.
        for i in range(_NPETAL):
            q = self._petals[i]
            if q[5] <= 0.0:
                if self.rng.random() < (0.5 if beat else 0.06):
                    self._spawn_petal(i, at_crest=self._w['wave'] > 0.4
                                      and self._swell > 0.5)
                continue
            q[3] -= step * 0.22                      # gravity on the petal
            q[2] += step * (self._wind_dir * 0.05 - q[2] * 0.5)
            q[0] += step * q[2]
            q[1] += step * q[3]
            q[4] += step * (1.4 + self._treble * 3.0)
            q[5] -= step * 0.16
            if abs(q[0]) > 1.1 or q[1] < -0.55:
                q[5] = 0.0

        # Drips are released from the crest on the beat and run down the sheet.
        for i in range(_NDRIP):
            d = self._drips[i]
            if d[3] <= 0.0:
                if beat and self.rng.random() < 0.30 and self._w['wave'] > 0.3:
                    self._spawn_drip(i)
                continue
            # A run stops when the ink is spent, rather than streaking off the
            # bottom of the sheet.
            d[2] = min(0.30, d[2] + step * (0.10 + self._bass * 0.22))
            d[3] -= step * 0.26
            if d[1] - d[2] < -0.55:
                d[3] = 0.0

    def render(self) -> None:
        """Upload the plate weights and everything placed on the sheet."""
        p = self._prog
        gain = max(0.0, float(self.parameters['neon_gain']))
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
        p['iNeon'].value = gain
        p['iWind'].value = float(self._wind)
        p['iSwell'].value = float(self._swell)
        p['iCrest'].value = float(self._crest)
        p['iCrestY'].value = float(self._crest_y)
        p['iSky'].value = self._sky
        for k in self._PLATES:
            p[f'iw{k.capitalize()}'].value = float(self._w[k])

        self._tbl_petal[:, 0] = self._petals[:, 0]
        self._tbl_petal[:, 1] = self._petals[:, 1]
        self._tbl_petal[:, 2] = self._petals[:, 4]
        self._tbl_petal[:, 3] = np.clip(self._petals[:, 5], 0.0, 1.0) * self._w['petal']
        p['iPetal'].write(self._tbl_petal.tobytes())

        self._tbl_crane[:, 0] = self._cranes[:, 0]
        self._tbl_crane[:, 1] = self._cranes[:, 1]
        self._tbl_crane[:, 2] = self._cranes[:, 2]
        self._tbl_crane[:, 3] = self._cranes[:, 3]
        p['iCrane'].write(self._tbl_crane.tobytes())

        self._tbl_drip[:, 0] = self._drips[:, 0]
        self._tbl_drip[:, 1] = self._drips[:, 1]
        self._tbl_drip[:, 2] = self._drips[:, 2]
        self._tbl_drip[:, 3] = np.clip(self._drips[:, 3], 0.0, 1.0)
        p['iDrip'].write(self._tbl_drip.tobytes())

        self._tbl_gear[:] = self._gears
        p['iGear'].write(self._tbl_gear.tobytes())

        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
