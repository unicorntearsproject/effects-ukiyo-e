# effects-ukiyo-e — Unicorn Viz effect pack

**Version 0.2.0**

Neon/steampunk ukiyo-e: a woodblock print that composes itself while you watch.

Each effect subclasses `unicornviz.effects.base.BaseEffect` and is auto-discovered.

## How the pack works

A woodblock print is built from **plates** — one carved block per colour, inked
and pressed onto the same sheet in order. This pack works the same way. Every
element is a plate with its own weight, and a director on the CPU brings plates
in and out over time, so the picture is never finished: it composes itself,
which is the part a static print cannot do.

It is the 1830s vocabulary run through neon and brass rather than reproduced.
Hokusai's claw-foam still hooks off the crest, but it is edge-lit; Fuji still
sits in its bokashi gradient, but there are circuit veins in the slope and
something burning in the crater; the cranes are clockwork; the seal glows.

**The plates interact.** One wind value drives the cloud drift, the rain's
slant, the blossom trajectories, the cranes' heading and the sway of every
drip, so they lean together. The wave's crest is a position, not a decoration:
blossoms are torn off *at* the crest when the swell is running, drips hang from
the wave's own edge (the shader evaluates it at each drip's x, so a run is
always attached to water), and the cranes lift as they pass over it. The beat
scatters the cranes, bursts blossoms off the crest, releases a drip and flashes
the seal.

**The director listens to the VJ system when there is one.** `VJApi.current()`
carries the song-structure hint auto-vj and the mixer publish — a phrase role of
HOLD/RISE/PEAK/FALL/CLOSE — and the printer works to it: a RISE inks plates one
after another, a PEAK has the whole picture on the sheet, a FALL wipes them
back, a CLOSE strips to the plates that carry the composition. Tempo comes from
the same place and turns the orrery. None of it is required — with no app
running the director falls back to its own timer.

Everything with a position — blossoms, cranes, drips, gears — is placed on the
CPU and handed to the shader ready to draw, so the fragment program stays a
rasterizer rather than a simulation.

## Effects

- **Floating World** — the sheet, the bokashi wash, drifting fans, kumogata
  cloud bands, Fuji, a torii, three wave plates with claw-foam, ink drips, the
  shower, blossom, clockwork cranes, and the printer's seal.

## Tweakables

`speed`, `zoom`, `reactivity`, `glitter`, plus `neon_gain` (how hot the neon
runs) and `layer_churn` (how readily the director re-cuts the plates; 0 freezes
the current picture).

## Changelog

- **0.2.0** — the brass orrery is gone; fans took its place. Gears were generic
  steampunk shorthand rather than anything from this picture's own vocabulary,
  and they read as borrowed. Fans belong here twice over: uchiwa-e were a whole
  print *format*, made to be mounted on rigid paddle fans, and a folding sensu
  can do something a gear cannot — it opens and closes, so the low end now
  works the fans open instead of merely spinning them. Both kinds are drawn:
  a sensu as the sector between pivot and leaf with ribs and guard sticks, an
  uchiwa as a round leaf on a handle. They ride the shared wind like everything
  else, so they cross the sheet with the clouds and the blossom rather than
  hanging in place the way the gears did.

  (Hokusai's *Watermill at Onden* sits in the same series as the Great Wave, so
  a water wheel is equally canon here and is still on the table.)

- **0.1.0** — initial release: Floating World.
