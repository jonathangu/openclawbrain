# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-234`
- winner mode: `graph_prior_only`
- trace hash: `sha256-80d3477f10050166bf08a79ad115cc0623875c77edbf3489b3449d2e77618193`
- fixture hash: `sha256-550621052f6f6f4dedd32e7dd1966df3bdae13f0842e74ffdcfed29aa308dfb9`
- score hash: `sha256-d8d2bf56b0a3db673146d08a6c240b319845a685741c54de159f03ab116dab7f`
- bundle hash: `sha256-977aed56342e05eaa57cab2f9ce965003a28e336abb57dcbba742865afbf78bb`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b053d8d2940510defb6223852e2cebf21b6ccd631a727caf5859e48b2c5a0baf |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7f9d43a946bc6e5f19d6c22f35d16da53008d7ca5b14bcc0bf544e50acc8b10d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e5c2ee2f292055224b32719acc47ab478388a143d5e76ae3b39cea6e7964e0bf |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-90a844e36a7a73892c47eb1ab1313dbc8f93752b28fbe6fcd67c5e7d7ecb4992 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2dce0404 | sha256-9d0239f8d76caa6e09349e171bc38ee5d3848937ccf0c79ef8ea13ef07379269 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2dce0404 | sha256-76c9114bc2d60c62c2f60f38f1e5503f8f5baeae8e137103a602676ed19fa473 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-22f257c7 | sha256-a614081fd6dff65a2ae079898bf6535385a39ddd4fdb0829bee39edaf7b2d6b3 |
