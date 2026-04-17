# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-058`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70e107aa90463a0c77bc30d344eca5153707641920ed24320747bbd52e05a0e6`
- fixture hash: `sha256-2a5cd5afc4b09fa9beced059043152cd23fab3958640aae8275a1e91138ba120`
- score hash: `sha256-e0525456990fc99cadf3bfd6bcd136214ca7bab6e69026660d266ac7ddb8b37c`
- bundle hash: `sha256-eb224c2f48576630b438bdb37a54b58bc8a5c09f091d51f9d8de14acd9479217`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1c146c3106aa3a476acb28a8b075ba9caa0dc741d245d11ea00bbf3c4bbed6c9 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6f4e14c88202b0dac3014488143d6b5300e67fd184c314fafecb6edee538038d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-71b9d474e1fe19308b7203b9a9042964f79a07e384d62806e312c924225ea5e6 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-e8c204ddb86b31c2d2599c3ff54ca4aba66844ea6c50a5ee5a88841abf729c2d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a617057b | sha256-d830e3f5aa3e8c9177c01c9b46508d9445330db869f6c0f005a6f3876e98f04b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a617057b | sha256-5757ddd4fcc574f6cf5a48bcb209964860867b29750921edde57dfe60f8ada24 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-6be33bbe | sha256-c6bfe5ee69cb2667141471a17df1073308ddbb342fd34252017c63067f19e13d |
