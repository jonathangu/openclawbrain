# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-db3086ba9270f5f47434d4a3c708e73ff2624adb056b71992e75ebf839a91592`
- fixture hash: `sha256-2a8d321cab2bd435ac998d63d68b17b7fce95e9a0ea6d02ef75e09676d4240bd`
- score hash: `sha256-c8f60dc0d05f558cc940ad43083db7da3e6224bdae4ec9aebfc42a97431c4814`
- bundle hash: `sha256-2019dc3d6ebf9dbdba0b6131eec51af14541cf7314be6fbd443cd1e98c442d7e`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-da6633515ce74e28e9f8bbc2cc587b6b0548deffeab6470c77c67fc675828106 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-04dcb770b3581b071235eee9eeeab36633a2adbf21184df1c60d36e89cad51f7 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9aae9f9c38ec3d8d8a560c31b0166b97a78e9d1d0126348030550366658f7a47 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-8c004b63bee0929e2daa77ebb9a5b74f7e47b858a19c4e92ec2851bed03fa8c9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0bc6f111 | sha256-0248743fdd8c91e820696bbe7e7a774ebff064f8fdd2b77c372d3145b216b1be |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0bc6f111 | sha256-5779c635318ab9d8a9245a1d0bf68c858f25b1b86c5859c07fc6a096502856cf |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-28eadcae | sha256-f5b0e720b8f44c3b46fdb7fe8c762fde249ce3b6c199b04a884ded90c95e280c |
