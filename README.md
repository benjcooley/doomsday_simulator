# ☄ DOOMSDAY Simulator

A real-time **WebGPU planetary-collision sandbox**. Earth, the Moon, and the solar
system rendered from live Keplerian orbits — then you throw worlds at them and watch
what happens. Each body is a **self-gravitating blob of ~10,000 GPU particles** with
deformation, melting, vaporization, heat conduction, radiative cooling, and ejecta —
so collisions excavate craters, splash, char, and cook a planet to death instead of
just playing a canned animation.

**▶ Live demo: https://benjcooley.github.io/doomsday_simulator/**

> Requires a WebGPU browser (Chrome/Edge 113+, Arc, or Safari 18+ on recent hardware).

## What it does

- **Live solar system** — real JPL approximate orbital elements, date-driven; planets,
  Sun, asteroid belt, Saturn's rings, starfield. Pick any date.
- **Particle-blob bodies** — Earth/Moon/impactors are thousands of particles with an
  iron core, rock mantle, painted continents, oceans (real fluid), and ice caps. They
  deform, splash, and gravitationally re-accrete.
- **A real thermal model** — impact friction + shock heating, conduction (magma smooths,
  ejecta scorches where it lands), an atmospheric heat reservoir that bakes the globe,
  exposure-based radiative cooling, and a physically-shaped incandescence gradient
  (ember-red → orange → yellow → white-hot).
- **Live readouts** — humans alive (tracked from livable land), surface & atmospheric
  temperature, Earth mass, Moon status, energy released (in Chicxulubs).
- **HDR + bloom** pipeline with blackbody emission for the orange-glowy carnage.

## Scenarios

Moonfall · Smash Test · Fastball (Moon at 30 km/s) · Theia II · Hit & Run · The Lance
(a 500 km asteroid at 0.1c) · Mars Attacks · Venus Descending · Pluto's Revenge · Moon
Billiards · Ring of Fire · The Embrace · Chicxulub XL · Apophis 2029 · Jupiter Drops By
· Comet Barrage · Project Icarus · Mirror Earth — plus an **Impactor Lab** to design
your own (size, speed, angle, target city, or raw state vector) with a live trajectory
preview.

## Controls

| | |
|---|---|
| **Drag** | orbit camera |
| **Scroll** | zoom |
| **Time dial** | slow-motion ↔ fast-forward (drag left for bullet-time) |
| **Space** | pause · **[ ]** halve/double warp · **H** hide all UI |
| **Focus** dropdown / click a label | follow a body |
| **Sidebar** | Scenarios · Impactor Lab · View options |

`?scenario=lance`, `?q=low|med|high|ultra` (particle count), and `?hud=0` URL params.

## Run locally

Any static server works (it's plain ES modules — no build step):

```bash
python3 serve.py 8413      # then open http://localhost:8413
```

## How it works

One fused O(N²) WGSL compute kernel does pairwise gravity + DEM contact
(spring/dashpot/cohesion) + collision/shock heating + melt-dependent material behavior,
per substep. A double-precision Keplerian layer provides the orbital backdrop in an
Earth-anchored local frame so fp32 stays precise. When the dust settles, mechanics
sleep and only an O(N) thermal kernel runs — so you can fast-forward days and watch a
dead world cool. See [`HANDOFF.md`](HANDOFF.md) for the gory engineering history.

## Credits

- Planet & star textures © [Solar System Scope](https://www.solarsystemscope.com/textures/)
  — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Pluto texture procedurally generated.
- Ephemeris: JPL "Approximate Positions of the Planets" (Standish).
- Built with [Claude Code](https://claude.com/claude-code).

## License

MIT (code). Textures under their respective licenses above.
