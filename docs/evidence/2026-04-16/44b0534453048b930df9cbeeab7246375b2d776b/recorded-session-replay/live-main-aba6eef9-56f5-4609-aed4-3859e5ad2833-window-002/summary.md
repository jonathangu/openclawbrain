# Recorded Session Replay Proof Bundle

- trace id: `live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ca1018694169cc3fe531485fc537c09a6239e84c0e4410a019dba97e2a66fe7e`
- fixture hash: `sha256-9d6b96efb0f7a7d48de55af286c816bef6a9a27fdc8a979e0eeba28c500d12da`
- score hash: `sha256-2e8bca19aaea5ace01864a96c521709322f43efe42b82c2eb35815f20a52e4c8`
- bundle hash: `sha256-0b04794814e898041a82bd6e3530d7beb013fc96790a8a600355eca943b13e97`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2225486ce356841ccd69a322b5b86cae51f3de0b57802b050f099b2bdb0a0f2e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dd1bc6dabfbc76d305bed01db1f0b3706757e98bc67b63ecf94f729ffcbef776 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b3b17d901792b407fecf3440dd8bb6b24f34408e5493c2dbd361e2dcb6ff0d7a |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-485916a74699f698ad82bb76e32a2612d918d86000cf4f5b259e076fc16e902b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b50cf53d | sha256-389ecbdcb3e354249179ec5c58fe5ff1818e5c1b4a02af4b62415edc94544ca2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b50cf53d | sha256-3dcf20f7c40d15ddaaa2c0ce0d20b6754e273a788ca71e496c5cbb638b2e3af2 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-b4cb3dac | sha256-7301a9609d31e9fd6533820412a486ba5466919a37c2d5242c80a88305c4a9b5 |
