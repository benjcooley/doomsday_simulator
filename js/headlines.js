// headlines.js — global rotating "news ticker" pool, fired by doom phase regardless of
// scenario. Tone: realistic-but-funny, deadpan gallows humor — plausible real headlines
// with a dark twist, not zany. Scenario-specific headlines fire on top of these.

export const FILLER = {
  // calm: nothing officially wrong. Faint institutional unease.
  calm: [
    'Bucketlist.com becomes world’s most-visited website',
    'Therapists report nationwide surge in "vague sense of dread"',
    'Astronomers "monitoring an object," decline further questions',
    'Defense department quietly cancels all leave',
    'Jewelers baffled by spike in last-minute weddings',
    'Vatican requests private audience, unusually quiet',
    'Survivalist forums hit record traffic, mostly lurkers',
    'Search trend rising: "how far is New Zealand"',
    'Markets close flat; analysts describe mood as "off"',
    'Major observatories stop returning press calls',
    'Pre-orders open for backyard bunkers, ship date "TBD"',
    'Sales of telescopes and rosaries both up sharply',
  ],
  // panic: a threat is confirmed and inbound.
  panic: [
    'Pope: "God isn’t returning my calls"',
    'Bucketlist.com crashes under record traffic',
    'Markets suspended: everyone selling, no one buying',
    'Flights to New Zealand sold out through next decade',
    'World leaders disappear from public view, no explanation given',
    'Gas stations empty as nation collectively "drives somewhere"',
    'Churches, mosques and temples report standing room only',
    'Mortgage payments mysteriously stop nationwide',
    'Tech CEOs photographed boarding private submarines',
    'National hotline for "the end" overwhelmed, please hold',
    'Grocery shelves bare; nobody bought the kale',
    'Stadiums fill for "one last game," scores no longer kept',
    'Out-of-office replies now read "permanently"',
  ],
  // doom: catastrophe underway.
  doom: [
    'Emergency services overwhelmed',
    'Power grids failing region by region',
    'Final broadcasts urge public to "be with loved ones"',
    'Air traffic control signs off: good luck!',
    'Looting reported for Beanie Babies, vintage Pez dispensers',
    'Scientists’ last paper titled "We Told You So"',
    'Stock prices frozen at the last closing bell',
    'Internet outages reported; doomscrollers panic',
    'Insurance industry: impact is not covered',
    'Hospitals shift policy to "comfort only"',
    'News goes dark as reporters join family',
  ],
  // aftermath: it’s essentially over.
  aftermath: [
    'No further updates expected',
    'Bucketlist.com offline; all items marked complete',
    'Satellites stops transmitting',
    'Geological record gains a notable new layer',
    'Surface conditions no longer surveyable',
    'Final weather report: extreme heat',
    'Census counts to be revised downwards',
    'Mission control: silence, then static',
    'Government reports: there are no secret escape rockets',
    'For the record: it was, in fact, the big one',
  ],
};
