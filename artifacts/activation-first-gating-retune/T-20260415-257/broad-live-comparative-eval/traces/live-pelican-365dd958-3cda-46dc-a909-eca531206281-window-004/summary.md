# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-365dd958-3cda-46dc-a909-eca531206281-window-004`
- winner mode: `learned_route`
- trace hash: `sha256-414029967a4dfaeacf3048f9cc246c927617fc5206e50ba6c1c2944d9dd8d93c`
- fixture hash: `sha256-2f96d4d80b85de0482bdf816d900c02ecf0137642687879ce902112bb8056ccc`
- score hash: `sha256-4ef5dbbca5b73eee2cf06b44a2a84adb3c2d602d25ad74ee07f0346590d3464c`
- bundle hash: `sha256-b40d317b55f474898edb7722c7823668e9b9ed84fb04f7814798be778f66cef9`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 60 |
| 2 | vector_only | 60 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d02c67e07266eced41424ec8d8650df73f7c0173cd9e14609381c09dbbd89d1f |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-37ef437d5872c9bfc18cd98183a300cc385668667d8c46d73b8b0d1b3a2bcb64 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-575bdd9395f2a5ae655ed5740a2796f495ae90a82ac67f30d67b029183dbca89 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-7cf83aaca15022765ba1fd3c22aef1300e27cd5954dd984ee7627d4c95044d98 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-f41fce84 | sha256-9fb873575b7e14a0bb2bd8905bba734d20e84092c24ec70cb42f7066337645bb |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f41fce84 | sha256-bbe0d1ceeea4bdb3097a87c72bf1147e6e4d72a566d954d62ff49ee9b2c27994 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-9bbc2f77 | sha256-c78b40c10735547c93fcb8caf88061e9d27ec264357df892a2af48f0bbcedec9 |
