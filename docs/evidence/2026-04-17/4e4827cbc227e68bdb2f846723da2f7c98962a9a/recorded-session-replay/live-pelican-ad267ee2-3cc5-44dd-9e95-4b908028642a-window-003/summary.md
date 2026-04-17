# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-154227e12deeada99188001de1f98c7859b44b0240a0a63280198f0600727836`
- fixture hash: `sha256-141e98c67b76e6b544c136b2dc9ec311316dae947241f48af13f9b3f509e9c48`
- score hash: `sha256-0cb73e8dc8501dcfbe2e99326e662920f4dce331f54f5b3db7412597612858ef`
- bundle hash: `sha256-0ba3ce7dc96a46aa3a223c3ac43f5403ce1487b62df04cbe3b2fd6b91bfff510`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/12
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 1 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5fd20c45ec549a50541ad825ca2263c2905bab11bd8f991e3eba1789bd6eddad |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-be564b2a3355d90f0294f3468346cfdb4780c2002f32ae0d60ff164019fc7318 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5ca1ba78b7734d561d29dac96b7c8e0dcf1dc4f3237d78562a1861021cf0f01e |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-1abd20a5ffc75687067a1e417d14d2a7ad485392134737b5a6bf76989a35d8f2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-60e56170 | sha256-2a8181430ab79e88c1ac9a4ae62660679186afc562859fd4482158859513987b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-60e56170 | sha256-680a015f4ebfcc3153a4113c5180575b45723794f7f68faea3e9d3eeb5734ec7 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c7147297 | sha256-a20880fe40417ddb6fce543853768fa674fe614fd16062b23646dcc425eb4c0d |
