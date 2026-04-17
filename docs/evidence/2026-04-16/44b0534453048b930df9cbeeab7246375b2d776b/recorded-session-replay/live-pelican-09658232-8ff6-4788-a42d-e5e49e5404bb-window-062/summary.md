# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-062`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68dd4afa8cf6968b41418bd460fed5641fe37e0c30a004be1adb6fd97d678410`
- fixture hash: `sha256-0984347c035679f491e5e5ce92160de0970752142af6bd7d0f80779707ccfa84`
- score hash: `sha256-1ca1b78230f8abdc720f76f16f6b68d5099e78bb30c8250d8ba7e6da5b5466c6`
- bundle hash: `sha256-3caee8ccbacc213d0615d3d5ec5cd2273c3c7c6f4b67f989fabcabd5b46c1e09`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4e7043d44034a818b042ec107f761c7c9e4d805591027e32242d8b764dc9d866 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c41dd62d20ed338d8384cc6f7628994007015aad26878bbb3faa662a0a518d17 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-07fa526ca18fe40db66d613b0763fb5d6882512bd877d927e84cdcd450df0257 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-f173990a9b88b2bb68925d88517bb9d55a3c718e5bb82f6e998e79886ba1d1b8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-20eb859c | sha256-297b8e4e8b34f5b942e8b275afd60ef453a82020ca04b527e89ee1c76769fbaa |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-20eb859c | sha256-12fac2d3ad12e5674bd8169258eb9055120c35c52d6764d75d63399c921664b8 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-2f051cf9 | sha256-c50c3ba8bf2b0614e36e11274cd0678ffb2cb67bad3a81147c3747f1b7fa456e |
