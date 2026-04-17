# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-034`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b1cc11a021b01ca8aba99240d6af1922e93eb57eb02a859f34b6e95e9d51abb2`
- fixture hash: `sha256-25597cd73d9bd0f0d086440e05e70594b904a1395b870c35b683a0a720d202a1`
- score hash: `sha256-466a86f60fa04faad2ea5977faf8971f3ad59894ea453e96e56c310bf86925f7`
- bundle hash: `sha256-3f9cd64e7bd27d9816528f77ba3ddda6a1118c73b825404748d535a5f3aec096`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8e0e779f7703426d9dc6e56462e7c187c6dea02bfd7bb266c409658015b58695 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ad54442cd1bc52281dd41eefe9f9d598e0e49af3bce10e28deb978a5fd013e27 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ac94b0be7f9502113ff564279a046435f3fc53c94bfa61f0412612e2724a1e07 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-84d47efa866f3441cecb2e88eee19e61c2730b101be297a4a7395b7f8f59e346 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6b379518 | sha256-6f99107e7945a270f94d5adc2d465518fe82a2d93d5894a4938739934627a3ff |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6b379518 | sha256-124bc60c8356c2eafa9a0422cabda740b7e3d0c2f75a70a132c98a4013fbb3d6 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-3641fc17 | sha256-be28457ee983bdfb08bcf441810d70081dc0e72c7fe8173d92583dedddc6c2df |
