# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-027`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3f260af2c7b68b1309e9a87df75f2e99f6d28d47bb3f82fdbd20cd787e51e3c0`
- fixture hash: `sha256-4a50ee1d4a23bf54584481d6c799516fa1f1a51aa4c19299da0f6a6b73848dff`
- score hash: `sha256-2ebb022b0129abbd8ac1fe527b2505c87f855c0d7c9deee53fc53428a7610094`
- bundle hash: `sha256-0164ba5569d109fc1b98b742ddaad349f7e5159f9e15278234055ab04461beb3`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9b2a597464226db9617a3470772ef24fd543ab0477b7bbc0a0ad5adf41bc0dc2 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1cd6008a951e78dbacbae3831d788bb087d0899506bcacaab26d3dd92479763f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-850421dd839d8d1541ac966450690815ad842463720fd7a77b12692333f7e230 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-4b864b2dda32004b6bd9f52229b26e01867863df307179b426ec22aa63367099 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1ae2fc1b | sha256-70670e9b2cbdfdbbfbeb68c744bee941ad56d6dcf0eab9c9dfc6a81e43f14c6b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1ae2fc1b | sha256-7d9ae9446cac6bfb707bcdba2d6e8cd60b630e1bcc596b54c8f282b27f480400 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-02c012d2 | sha256-da1981c6e1a01e84a49a0910ed4d3c8920e09f37013f0719ec0908f08b1b980b |
