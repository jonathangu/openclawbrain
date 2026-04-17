# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-164`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f337daaeed6bc47fc68765c0195f530bb4ce38ec076e00ac4c73412b426d85da`
- fixture hash: `sha256-5e5cab708ce5b294bac69d34a6279b47e648ad8d40ed85f35998caca6e589c7b`
- score hash: `sha256-625e8dbb42d2e44dadd6721983dc9ac1687719df22bbe89005d8a6857242421a`
- bundle hash: `sha256-fdb8bf4459db348b7186a167040bf620051bdf8d86a19052539cb059a37fc7cc`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-100c819b269094add6922ee0aca0d157fd41366c476c3703f8d276f1431d3315 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ec39105799212a921851b6a91a0722f78fe2fd844640a8f37a90e2c626693fe3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8b213e5c9ad9ac3fa0d6e38aea5b95e426cad4c531789b7a18b96e80e9e54cfa |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-141d8adb752af4a88061e2d90660237d178f2ddd23df84340d8836a9ebd7e335 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ad33108c | sha256-11ff010d7b4747d5de319b31fcc8c34e2ae2eef614fb19adf9852c97461098bf |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ad33108c | sha256-ebff294f13d23ef3dd7c68ba5aa10f9486253587f483af5c46c66d565dd52fa2 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-37dc319b | sha256-570eb9a79e6cd0d8cfd8605425528161938c2cc3d693b52858902d7edecef096 |
