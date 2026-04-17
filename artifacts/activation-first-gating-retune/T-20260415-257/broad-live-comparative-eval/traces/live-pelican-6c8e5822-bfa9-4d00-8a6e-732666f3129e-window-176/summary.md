# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-176`
- winner mode: `graph_prior_only`
- trace hash: `sha256-480f373d763bedd2f5766cb9a1a8860701112223bd910911c4830c2fc4277912`
- fixture hash: `sha256-2055d04a6856d7cf43d112e858be3651b8402ee14faf73331e6f59144245384e`
- score hash: `sha256-033ad40d39920203d9a4b4c32e92a986b3aa59fedd666d5458fd37784605c29c`
- bundle hash: `sha256-ef9fbf161b1b17f795c0c60645b485bde865bad90f08c5b41b47b783a622046b`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cc6a8a76ddb7a25937feec38e19ee175087db4867c24970d2759b39f1c9b4bd1 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1ea986d45104fe01adbbf65e988a4046b0c5fa318ee8ad19147076a32922b2b4 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9b9059781a2a06799f66f68adeb7017f63d9054089328c1a341247cbe52dd66a |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-658829957466dbcd9a6343370610a1c1686921e82635355728c32f8e43cebd1f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-61978df4 | sha256-1366c43f9602e0fac8f906c7fb4de19762bfdf1a793603757c4f3c22c970f884 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-61978df4 | sha256-299f1e0979c963207a74eba6a4c363eebe8972e5e24da2d794fb687cc10433c6 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-462cdcbf | sha256-e025e4891277d61cf73a126ba79ad36d3a3472a7dd691b48839a3a29fbe0f5a0 |
