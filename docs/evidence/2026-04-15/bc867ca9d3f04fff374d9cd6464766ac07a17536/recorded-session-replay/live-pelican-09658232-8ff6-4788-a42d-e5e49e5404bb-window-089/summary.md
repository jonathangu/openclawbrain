# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-089`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7a2236f637704fe149867dcf144d671dc7a13fa94e04f98252bb7a94efde6a70`
- fixture hash: `sha256-3dd95fcccf0fb105acb53dbd74c41b44d30300251f8ca1b0c6b6f7ee328de982`
- score hash: `sha256-6cfbfccad4a512c7d6856009bbe75089b11121c064e521e4cb4c3d11e233d7cb`
- bundle hash: `sha256-b106394cb0cca50dbd9181d55e49722654ea184d42d7b79462596e0a568543aa`

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
- phrase hits: 0/4
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ef52e9f6d08b86a0755671620744d8fa71177a56d88b43c65d023da00ed4b3db |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-473d8e323d157559e2840107e7d45d2da3c01e72d6e32d1a130e21810d9451b4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cae51ccf54598d367b8cad18aabf257d51bfbcabd69f239be6766cd42039f961 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-2336721c29be0c76afbb53d3eb01014bea313d1720bb601e7089d40f44d86825 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-79c0bee5 | sha256-c16861f368bdfaf22f5bb47e404f9256244cec3d88baf92ec225487aea2c9400 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-79c0bee5 | sha256-8032da7570ec8e30e975e1505cf431c12e7938df21c3bce221daf826d2c13a48 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-79c0bee5 | sha256-c16861f368bdfaf22f5bb47e404f9256244cec3d88baf92ec225487aea2c9400 |
