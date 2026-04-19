# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-081`
- winner mode: `graph_prior_only`
- trace hash: `sha256-404edb8e8148990f7dd9ba6ee7f25c05f0fb22e6cb89bbcabd63ecc3b578e01f`
- fixture hash: `sha256-1b74e2bda5a418c592958218e18909375ff9b60c95cbb866c13aa0c8c5768f8f`
- score hash: `sha256-b66e90bcdc75a3dc1b6cc5f3d76b32b8c14d39e7b85fe845b72b11b1181a40e8`
- bundle hash: `sha256-f8618aeb0532fdd3caf54e99e69b452724cd359e846597de7af2baad59728c31`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d8d61d87f3931e7b37ce8220e425e901452b419dbbce6c76196690cf92892dce |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-f602e5ab60aedf10d325606be5dbd91ba7c767ad5b48fea474b4127de7a91798 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-5b5ef19d945b95580ed3df58479a10d56abb36e2549a73a4b64a2a6f0335469e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-7ffc76be8ae5c6088765fd4d335c75a71679ff38f255276ab65e6a915b69625c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-ed08939d | sha256-cf4f12ce6a94cbab087e74d812e5882f46c2c6cb7fbe9c931304eed238e38b9f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-ed08939d | sha256-b154893b5ac7cf243a958a230a064f3c75bd4160fa0a3880e986d30f3bf26524 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-ed08939d | sha256-cf4f12ce6a94cbab087e74d812e5882f46c2c6cb7fbe9c931304eed238e38b9f |
