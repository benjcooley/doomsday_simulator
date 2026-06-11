// scenarios.js — the doomsday menu. Each scenario configures bodies, aim geometry, warp & headlines.
// ctx helpers (provided by sim.js): spawnImpactor(opts), dirFromAngles(azDeg, elDeg), jd0, LD (lunar dist Mm)

export const SCENARIOS = [
  {
    id: 'smash', title: 'Smash Test — Moon → Earth', skulls: 5,
    blurb: 'The Moon, dead center, gentle merger speed. Watch the goop, the ejecta rain, and the planet cook. (Press H to hide all chrome.)',
    warp0: 1200, focus: 'earth', camDist: 200, moon: 'none',
    view: { orbits: false, belt: false, trails: true, autoFrame: true },
    build(ctx) {
      const dir = ctx.dirFromAngles(18, 7);
      const { pos, vel } = ctx.approach(dir, 140, 0.008, 0);   // gentle hit: 8 km/s approach
      ctx.spawnImpactor({ recipe: 'moon', d_km: 3474, pos, vel, name: 'Moon', countScale: 3 });
    },
    headlines: [
      { t: 1, text: 'Smash test armed — Moon inbound, dead center' },
      { cond: 'approach', text: 'Moon closing fast…' },
      { cond: 'contact', text: 'CONTACT — let’s see what happens' },
    ],
  },
  {
    id: 'fastmoon', title: 'Fastball — Moon at 30 km/s', skulls: 5,
    blurb: 'Same Moon, 2.5× escape velocity: full excavation regime. Craters and ejecta curtains instead of a gentle goop-merger. Compare with the Smash Test.',
    warp0: 1200, focus: 'earth', camDist: 200, moon: 'none',
    view: { orbits: false, belt: false, trails: true, autoFrame: true },
    build(ctx) {
      const dir = ctx.dirFromAngles(18, 7);
      const { pos, vel } = ctx.approach(dir, 200, 0.030, 0);
      ctx.spawnImpactor({ recipe: 'moon', d_km: 3474, pos, vel, name: 'Moon', countScale: 3 });
    },
    headlines: [
      { t: 2, text: 'Moon inbound at 30 km/s. This one is NOT a gentle merger.' },
      { cond: 'contact', text: 'EXCAVATION-CLASS IMPACT — crust departing at orbital velocity' },
      { cond: 'after:7200', text: 'Debris curtain circling the globe. New ring system pending.' },
    ],
  },
  {
    id: 'peaceful', title: 'Blue Marble', skulls: 0,
    blurb: 'Earth, Moon, Sun. Nothing is wrong. Yet. Pick a date, spin the planet, enjoy it while it lasts.',
    warp0: 60, focus: 'earth', camDist: 30,
    headlines: [
      { t: 5, text: 'Scientists confirm: everything fine, weirdly' },
      { t: 60, text: 'Astronomers report "uncomfortable silence" from deep space' },
    ],
  },
  {
    id: 'moonfall', title: 'MOONFALL', skulls: 5,
    blurb: 'The Moon’s orbital velocity is set to zero. It begins to fall. Arrival: ~4.8 days. Tides get… dramatic.',
    warp0: 30000, focus: 'earth', camDist: 900, moon: 'stopped',
    headlines: [
      { t: 2, text: 'BREAKING: Moon decelerating — physicists "deeply unhappy"' },
      { t: 3600, text: 'Tide tables cancelled indefinitely' },
      { cond: 'approach', text: 'Moon visibly larger tonight. Do not look up.' },
      { cond: 'contact', text: 'THE MOON HAS ARRIVED. Thanks for everything.' },
    ],
  },
  {
    id: 'theia', title: 'Theia II: Moon-Maker', skulls: 5,
    blurb: 'A Mars-sized protoplanet on a grazing trajectory — the same impact that made the Moon 4.5 Gyr ago. Watch a debris ring form.',
    warp0: 1200, focus: 'earth', camDist: 220,
    build(ctx) {
      const dir = ctx.dirFromAngles(35, 8);
      const { pos, vel } = ctx.approach(dir, 160, 0.0042, 4.0);  // dist Mm, speed Mm/s, impact param Mm (grazing)
      ctx.spawnImpactor({ recipe: 'mars', d_km: 6779, pos, vel, name: 'Theia II', countScale: 1.0 });
    },
    headlines: [
      { t: 2, text: 'Object inbound. Mass: planetary. Mood: ominous.' },
      { cond: 'contact', text: 'GIANT IMPACT IN PROGRESS — new moon ETA: 100 years' },
      { cond: 'after:7200', text: 'Earth now wearing a debris ring. Saturn unimpressed.' },
    ],
  },
  {
    id: 'chicxulubXL', title: 'Chicxulub XL', skulls: 4,
    blurb: 'The dino-killer was 10 km. This one is 120 km, aimed at the Yucatán again — for tradition. ~2,400× the original yield.',
    warp0: 600, focus: 'earth', camDist: 40,
    build(ctx) {
      ctx.launchAtLatLon({ recipe: 'rock', d_km: 120, lat: 21.4, lon: -89.5, speed: 0.024, entryDeg: 45, azDeg: 60, range: 90, name: 'Chicxulub XL' });
    },
    headlines: [
      { t: 2, text: 'Yucatán braces for unwelcome historical reenactment' },
      { cond: 'contact', text: 'IMPACT. Sky scheduled to catch fire shortly.' },
      { cond: 'after:3600', text: 'Global dust layer forming. Sunsets: spectacular, ominous.' },
    ],
  },
  {
    id: 'apophis', title: 'Apophis 2029', skulls: 1,
    blurb: 'April 13, 2029 — the real flyby, 38,000 km out (inside satellite orbits!). Watch it thread the needle… or press the red button in the Lab.',
    date: '2029-04-13T18:00', warp0: 600, focus: 'earth', camDist: 130,
    build(ctx) {
      const dir = ctx.dirFromAngles(205, -3);
      const { pos, vel } = ctx.approach(dir, 420, 0.00584, 48.3);
      ctx.spawnImpactor({ recipe: 'rock', d_km: 0.375, pos, vel, name: 'Apophis' });
    },
    headlines: [
      { t: 2, text: 'Apophis inbound for historic close shave' },
      { cond: 'approach', text: 'Closest approach imminent — satellites politely step aside' },
      { t: 86400, text: 'Apophis departs. See you in 2036, you absolute menace.' },
    ],
  },
  {
    id: 'jupiter', title: 'Jupiter Drops By', skulls: 5,
    blurb: 'Jupiter arrives at one lunar distance. Earth and Moon discover what "Roche limit" means, intimately. Spaghetti time.',
    warp0: 2500, focus: 'earth', camDist: 1600,
    build(ctx) {
      const dir = ctx.dirFromAngles(140, 12);
      const { pos, vel } = ctx.approach(dir, 2600, 0.009, 320);
      ctx.spawnImpactor({ recipe: 'jupiter', d_km: 139822, pos, vel, name: 'Jupiter (rogue twin)', countScale: 1.0 });
    },
    headlines: [
      { t: 2, text: 'Very large object inbound. Telescopes refuse to elaborate.' },
      { cond: 'approach', text: 'Jupiter II fills the sky. Oceans leaving early.' },
      { cond: 'contact', text: 'EARTH IS BEING ACCRETED. New address: Jovian cloud deck.' },
    ],
  },
  {
    id: 'lance', title: 'The Lance — 0.1c', skulls: 5,
    blurb: 'A 500 km iron asteroid at a tenth the speed of light. ~10³⁵ joules — a thousand times Earth’s binding energy. You get about thirty seconds to watch the glint grow. (γ=1.005: the relativity tax is modest. The apocalypse is classical.)',
    warp0: 90, focus: 'earth', camDist: 140,
    view: { orbits: false, belt: false, trails: true, autoFrame: true },
    build(ctx) {
      const dir = ctx.dirFromAngles(95, 22);
      const { pos, vel } = ctx.approach(dir, 80000, 30.0, 0);
      ctx.spawnImpactor({ recipe: 'iron', d_km: 500, pos, vel, name: 'The Lance', countScale: 40 });
    },
    headlines: [
      { t: 2, text: 'Object detected at 0.1c. There is no plan for this.' },
      { cond: 'approach', text: 'It crossed the Moon’s orbit in twelve seconds.' },
      { cond: 'contact', text: 'PERFORATION. Entry wound. Exit wound. Planet pending.' },
    ],
  },
  {
    id: 'hitrun', title: 'Hit & Run', skulls: 5,
    blurb: 'A Mars-sized body at a steep grazing angle — rips a gouge through the mantle and keeps going, trailing both planets’ guts. A real planetary-science category.',
    warp0: 1500, focus: 'earth', camDist: 240,
    build(ctx) {
      const dir = ctx.dirFromAngles(210, -6);
      const { pos, vel } = ctx.approach(dir, 180, 0.013, 7.6);   // graze at ~1.2 Earth radii
      ctx.spawnImpactor({ recipe: 'mars', d_km: 6779, pos, vel, name: 'Hit-and-Runner', countScale: 1.0 });
    },
    headlines: [
      { t: 2, text: 'Inbound body will only "graze" Earth, experts say reassuringly' },
      { cond: 'contact', text: 'GRAZING IMPACT — it’s tearing out the mantle and LEAVING' },
      { cond: 'after:7200', text: 'Thief planet escapes with several trillion tons of Earth' },
    ],
  },
  {
    id: 'billiards', title: 'Moon Billiards', skulls: 5,
    blurb: 'Ceres slams the Moon at 80 km/s — hard enough to knock it out of orbit and onto US. The cosmic trick shot. Called pocket: Earth.',
    warp0: 2000, focus: 'moon', camDist: 60,
    build(ctx) {
      // hit the Moon's leading face to cancel its orbital velocity → it falls
      const moonNow = ctx.predict('Moon', 0);
      const moonSoon = ctx.predict('Moon', 1400);
      const lead = [moonSoon[0] - moonNow[0], moonSoon[1] - moonNow[1], moonSoon[2] - moonNow[2]];
      const ll = Math.hypot(...lead) || 1;
      const ahead = lead.map((x) => x / ll);
      const pos = [moonSoon[0] + ahead[0] * 110, moonSoon[1] + ahead[1] * 110, moonSoon[2] + ahead[2] * 110];
      const vel = ahead.map((x) => -x * 0.080);
      ctx.spawnImpactor({ recipe: 'rock', d_km: 940, pos, vel, name: 'Ceres (cue ball)' });
    },
    headlines: [
      { t: 2, text: 'Ceres lines up the shot…' },
      { cond: 'contact', text: 'CLACK. Moon in motion. Trajectory: us.' },
      { cond: 'after:43200', text: 'The Moon would like everyone to know this wasn’t its idea' },
    ],
  },
  {
    id: 'ringoffire', title: 'Ring of Fire', skulls: 4,
    blurb: 'Twelve 50-km asteroids arriving from every direction within the same hour. Nowhere to hide — synchronized global bombardment.',
    warp0: 900, focus: 'earth', camDist: 90,
    build(ctx) {
      for (let i = 0; i < 12; i++) {
        const az = i * 30 + (i % 3) * 7;
        const el = [-40, 5, 40, -15, 25, -55][i % 6];
        const dir = ctx.dirFromAngles(az, el);
        const { pos, vel } = ctx.approach(dir, 150 + (i % 4) * 18, 0.030, 0);
        ctx.spawnImpactor({ recipe: 'rock', d_km: 50, pos, vel, name: 'RF-' + (i + 1) });
      }
    },
    headlines: [
      { t: 2, text: 'Twelve simultaneous detections. Statisticians: "that’s not random"' },
      { cond: 'contact', text: 'FIRST OF TWELVE. The sky is falling, everywhere, on schedule.' },
      { cond: 'after:5400', text: 'Earth now uniformly cratered. Very egalitarian apocalypse.' },
    ],
  },
  {
    id: 'marsattack', title: 'Mars Attacks', skulls: 5,
    blurb: 'The actual Mars, off-center at 15 km/s. The red planet becomes part of the blue one — gouging in sideways, with both planets’ mantles on display.',
    warp0: 1500, focus: 'earth', camDist: 240,
    build(ctx) {
      const dir = ctx.dirFromAngles(150, -10);
      const { pos, vel } = ctx.approach(dir, 200, 0.015, 5);
      ctx.spawnImpactor({ recipe: 'mars', d_km: 6779, pos, vel, name: 'Mars', countScale: 1.0 });
    },
    headlines: [
      { t: 2, text: 'Mars has left its orbit. It did not file a flight plan.' },
      { cond: 'contact', text: 'MARS IMPACT — the red and the blue making a horrible purple' },
      { cond: 'after:10800', text: 'Olympus Mons reported somewhere over the Pacific' },
    ],
  },
  {
    id: 'venusfall', title: 'Venus Descending', skulls: 5,
    blurb: 'Earth’s evil twin arrives — 480°C surface, sulfur clouds, and 4.9×10²⁴ kg of bad intentions, swinging in off-axis.',
    warp0: 1500, focus: 'earth', camDist: 260,
    build(ctx) {
      const dir = ctx.dirFromAngles(285, 8);
      const { pos, vel } = ctx.approach(dir, 220, 0.011, 3.5);
      ctx.spawnImpactor({ recipe: 'venus', d_km: 12104, pos, vel, name: 'Venus', countScale: 1.0 });
    },
    headlines: [
      { t: 2, text: 'Venus inbound, pre-heated to 480°C for your convenience' },
      { cond: 'approach', text: 'Two Earth-sized planets, one orbit. This ends one way.' },
      { cond: 'contact', text: 'VENUS IMPACT — the twins are merging' },
    ],
  },
  {
    id: 'embrace', title: 'The Embrace', skulls: 5,
    blurb: 'A second Earth drifts in at walking-into-each-other speed. No fireball arrival — just two worlds slowly, irreversibly becoming one. Somehow worse.',
    warp0: 3000, focus: 'earth', camDist: 320,
    build(ctx) {
      const dir = ctx.dirFromAngles(60, 14);
      const { pos, vel } = ctx.approach(dir, 240, 0.0028, 4);   // barely above mutual capture
      ctx.spawnImpactor({ recipe: 'earth2', d_km: 12742, pos, vel, name: 'Earth 2.0', countScale: 1.0 });
    },
    headlines: [
      { t: 2, text: 'Second Earth approaching very, very politely' },
      { cond: 'approach', text: 'They’re holding hands now (gravitationally)' },
      { cond: 'contact', text: 'THE MERGER BEGINS. Both worlds melting into the handshake.' },
      { cond: 'after:21600', text: 'One planet now. Twice the mass, none of the people.' },
    ],
  },
  {
    id: 'pluto', title: "Pluto's Revenge", skulls: 5,
    blurb: 'Demoted in 2006. Radicalized ever since. 1.3×10²² kg of icy grudge at 22 km/s.',
    warp0: 1800, focus: 'earth', camDist: 120,
    build(ctx) {
      const dir = ctx.dirFromAngles(310, 20);
      const { pos, vel } = ctx.approach(dir, 800, 0.022, 2);   // slightly off-center gouge
      ctx.spawnImpactor({ recipe: 'pluto', d_km: 2376, pos, vel, name: 'Pluto', countScale: 1.4 });
    },
    headlines: [
      { t: 2, text: 'Pluto spotted leaving Kuiper belt "with intent"' },
      { cond: 'approach', text: 'Pluto demands reclassification. Or else.' },
      { cond: 'contact', text: 'PLANETHOOD RESTORED BY FORCE. IAU unavailable for comment.' },
    ],
  },
  {
    id: 'barrage', title: 'Comet Barrage', skulls: 4,
    blurb: 'A fragmented comet train — seven icy impactors over three days. Shoemaker-Levy 9, but it’s our turn. Orbital pinball at its finest.',
    warp0: 4000, focus: 'earth', camDist: 90,
    build(ctx) {
      for (let i = 0; i < 7; i++) {
        const dir = ctx.dirFromAngles(80 + i * 9, -10 + i * 5);
        const { pos, vel } = ctx.approach(dir, 300 + i * 260, 0.045, i % 3 === 2 ? 9 : 0);
        ctx.spawnImpactor({ recipe: 'comet', d_km: 9 + i * 3, pos, vel, name: 'Fragment ' + 'ABCDEFG'[i] });
      }
    },
    headlines: [
      { t: 2, text: 'Comet train inbound: seven for the price of one' },
      { cond: 'contact', text: 'FIRST IMPACT. Six more on the conveyor.' },
      { cond: 'after:86400', text: 'Insurance industry declares force majeure, retires to bunker' },
    ],
  },
  {
    id: 'icarus', title: 'Project Icarus', skulls: 5,
    blurb: 'Earth’s orbital velocity is cancelled. We fall into the Sun — 64 days of accelerating regret, oceans boiling somewhere around week six.',
    warp0: 400000, focus: 'earth', camDist: 60, anchorVelScale: 0.0, moon: 'keep',
    headlines: [
      { t: 5, text: 'Earth off its rails. Sun: "come closer"' },
      { cond: 'hot:330', text: 'Equator uninhabitable. Real estate booms at poles.' },
      { cond: 'hot:373', text: 'OCEANS BOILING. Atmosphere now soup.' },
      { cond: 'hot:1500', text: 'Crust glowing cherry-red. It has been an honor.' },
    ],
  },
  {
    id: 'earth2', title: 'Mirror Earth', skulls: 5,
    blurb: 'An identical Earth, head-on, at escape velocity. The cleanest possible apocalypse — two worlds, one splash.',
    warp0: 1500, focus: 'earth', camDist: 260,
    build(ctx) {
      const dir = ctx.dirFromAngles(0, 0);
      const { pos, vel } = ctx.approach(dir, 220, 0.0112, 2.5);
      ctx.spawnImpactor({ recipe: 'earth2', d_km: 12742, pos, vel, name: 'Earth 2.0', countScale: 1.0 });
    },
    headlines: [
      { t: 2, text: 'Second Earth detected. It also claims to be the real one.' },
      { cond: 'approach', text: 'Mirror Earth raises identical objections' },
      { cond: 'contact', text: 'WORLDS COLLIDE. Winner: physics.' },
    ],
  },
];

export function getScenario(id) { return SCENARIOS.find((s) => s.id === id); }
