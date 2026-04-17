# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-570958465e5589f279bb28e850af48f0de0e358122b512d402db3214c7541c3c`
- fixture hash: `sha256-06808b26154de9486de3e390d83e02d5c54e1e0ca160f5f4c88501af04825dc3`
- score hash: `sha256-6ed1bc1446024b65ea6c1eb5cec66989ae8f8be4f0ca99cd35682d83846b0c9c`
- bundle hash: `sha256-ad25720b39cc0b58d9960765d41a7ea44f0f6adaa89812731bf644c1acf592e2`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dd27477bdd5733b8ef83edfac9b06aafa0bfaf3753550669b2a8358e4c2d729f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-82ecd3226052e9695f9a7d13b1279491c039727a912d9c4b4a8fdecf1462cc77 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-83d2c972e7d1ce66d3904a2668544447288af7fd6573a6be593f47af572fce44 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-8ecdcc696b58bb394ca784d954e82f442c7e7e2b004a66890947c52df4fe862d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-71015f19 | sha256-d434f393c90e597223aba8464fdbac8085a2bef743eb5506adb8d77c2e5a66a4 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-71015f19 | sha256-65f92a63d25dbdd9a38d468b729cf5a99dfe5c44194c5929faf2cbd5e9aab47f |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-8a1b6182 | sha256-ad4da6a43c1e618fd79253ef8a2fb7db8240b549d68a8cff00364417677a0522 |
