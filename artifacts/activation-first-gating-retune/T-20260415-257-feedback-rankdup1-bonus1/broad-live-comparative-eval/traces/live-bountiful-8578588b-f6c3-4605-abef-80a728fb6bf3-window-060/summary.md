# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-060`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ba906605b22dc94a498f1f2a524326ba3cc034e63e0a380bea5c3ad692ee02f8`
- fixture hash: `sha256-a13d04d7740d5d386089602f56e23b4bb142c5bed9f7073eeb6516366a131246`
- score hash: `sha256-0486e6353f12b64d44152babfa22305271572d26faac2962976450725016f757`
- bundle hash: `sha256-1acf5f9fa82ba027249c688a52f9044c6717b3d9b62c721db2fedc607ec2fa39`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-aec7ec8f6799a0ee01f6b4130aafafed76ac2a835827ba3dac19bfdac983b407 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-62bc4b912cb8f1e523dfaa968a4f32351cc4a15a3c963890d3d9426c7603e7a7 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-a7dc4a70499a590947fc8998fbfab1ed10ca14fe174b1912c358d4b2322ce33d |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-959454da257e3336e3246a68a99633d823d8d1757c6d55fcfbb923806a1e96df |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-7b4f5d2d | sha256-14e5d1a8f28bbad77281ae0d24d9c67dbcb14b829137249a4f46de431764b942 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-7b4f5d2d | sha256-14e5d1a8f28bbad77281ae0d24d9c67dbcb14b829137249a4f46de431764b942 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-7b4f5d2d | sha256-ef24ad21dfd9a0db1da6b7be05ac6393b1c8216e1cb40ec9e747fa52cd3bc7ad |
