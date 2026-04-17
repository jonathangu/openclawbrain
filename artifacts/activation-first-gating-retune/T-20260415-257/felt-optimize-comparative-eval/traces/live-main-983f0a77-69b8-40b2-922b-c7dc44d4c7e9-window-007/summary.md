# Recorded Session Replay Proof Bundle

- trace id: `live-main-983f0a77-69b8-40b2-922b-c7dc44d4c7e9-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-13518408454d88b3ad692b956343d851ffe682724dcc9ea68679835cb38cd6f1`
- fixture hash: `sha256-d8ddfc141ca061b024a7735fc1bd6c41a09ad3c89f85b7541ee5a4463459f049`
- score hash: `sha256-201ea2adc7690e0c2893c4cccb7f9e1f6bf488b57d0b0275c18e2a43031cbc9d`
- bundle hash: `sha256-26184db7e408c98b1c5a195ae419e62d09164bf9952451a9582391a32c3e93d3`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/12
- phrase hit rate: 0.083333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-76a9870f23308038c7dfa2834df546254ae4769b20da16b32ac7e7ef5f9b078e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c277084e9962faf097cc30351f4aff48ab9a6c9977337fb7e97effd7bffca301 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-5e6590d3acbb31614bed90fd19acf890fbd397570886a836fb4919b2bd710773 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-4cbeef2a0fb5d947935b65fe9ef8a74af487cde58e61f53af79dacae655a7ea3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9e7996ea | sha256-1aa48299887b294224d2c6260e3932dffe7f8186c51277def9cd70fe5a7a3c52 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-9e7996ea | sha256-18d6dff36bc130a1e46bba6ac63d8362e212cea6d1293d87bd3b0123c821c130 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d009b019 | sha256-d8799e8148dd1a3a727ce6672270934061c5b11483ec916d70b33021aba3038f |
