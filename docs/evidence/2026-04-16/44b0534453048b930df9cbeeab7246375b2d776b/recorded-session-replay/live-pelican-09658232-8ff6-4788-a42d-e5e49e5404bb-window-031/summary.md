# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7523c240f6bfd6f671f735664700ac6dc9001a24ca15ade4db3948ee85bd8854`
- fixture hash: `sha256-cfb2a9ca78d96cf92b242a10f940aec587e9ef8a10abc64c8e738df9cb79ae77`
- score hash: `sha256-fd768482bc3b7131a6c18eff2fc4074a79039504a5ce869ac7970d784ba3be96`
- bundle hash: `sha256-20028703b07459d290c71dd094324e31383b4c9eb4b6eb0c9c5985c274ba00a5`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e58229df13dcf750a6ed3af7bdcf953c6815a1f652d19ad1292758f1bf838de0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-71ff82a7ef9ec0e514e61e9d2eaff417a7983844f46cf4267a1ea7617ef78225 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-64799eb4904758e81d43a8cdc8ded8de1adf69a9237235fa58fbb9268366cc85 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-5b9b7df7197e876f51071f61484e9ced45d73e7cadb2ad11d033546fe49f4f9b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1e592ca8 | sha256-978f912ad9eb73c622f9595e2a90bdca040a3a57d459503aa07918463694c977 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1e592ca8 | sha256-926c11844d9bbadfda49f58914ee382ec3aa3a7180da4265e83a2cf6f3ad9dea |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-8c6c0a5b | sha256-bd3a65387dc3cde331ea2c14f183d29c82886f663a65f5df61acc464fe2b214c |
