# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-170`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b236d1de0cb5cdb5c6745d8cf8165eea05f61fe4f12fe91030959e1a1f0a9ef6`
- fixture hash: `sha256-2918c2c3bf776980cb54310652408dcd4b80904c74dc802c02149421011a5050`
- score hash: `sha256-d29d8dbe4543015f626cc47cdb10f5641ef94294e57d179f1b054487cce08f6f`
- bundle hash: `sha256-0c138a2282946c249a4131593aee29e3e09e5b896b5da702467fa6bf63b25a83`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b752caeb990c3acabff60b3183401c5659a9fe06fb13d30bacaaff23a3d4f453 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d7158def7921878ed5fa47f4daea3a6b30d540617324932876c14cda212695e7 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-15579c561665d9812c4038eec09792b98ceddb7c445b9f47a7ad293452584886 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-1ebd8464021a72f4c947755b38c2364726ba3146a9b9a9303c6fb997c0c10a53 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-462d99dd | sha256-8b8416e536be9200d836e1ee567fd36244081e71f3d9a9a72863bc15b6e65e8e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-462d99dd | sha256-7f3482f21d5e088a24553270572dd8c783aedfeb52ca71fd08c732f5e189e994 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-38e3da62 | sha256-c8d27455855adfc716e2f3e62ece10ff3ff908c84c2fecddcc21530e22996376 |
