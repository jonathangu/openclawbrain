# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-3cb1eac8-82e4-48bf-8898-e9fa5bead77a-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-79532f3b0ed4010e846f65618be48c2307af13b97dd581f294dd9d5e6325f8eb`
- fixture hash: `sha256-6cb2c0584a4478c43146057da013c5b788958c7763f4cd2e66653360656b5ed8`
- score hash: `sha256-8b714daed72371b0059c3c7daf6b46ee131c5a23e9929e2245785558de17037b`
- bundle hash: `sha256-4575747324d0a9639a42f1bfeebb3e4bed75a39aea45f5de23e9486258eeae08`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9133ee007dc762137868e0af1d2b1845ed239e3a386c6ccc9187f6b355e2ae22 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ef57bdbc9f30f9224fdaca12e0da3dd7e4a6a11d86369f3e99875ce0723a4efa |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2a65857ac9c26f00e8aa4149667829f426291cf130343ae7e2a1f2c9e43ffbdc |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-729f190d7073a1d01c7ae8011de102e65fdcbf26c50f44751e0300b013fad9a6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8853d946 | sha256-c642917b052a158134e7713d7907ef3df9ff59e673201fb1855e20b0096dc817 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8853d946 | sha256-20c415590c1f9cea3046d0f4e0902c762b5274d3340fdb0c72c9690789984959 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b02088e7 | sha256-c1e222b8550a77b00232772f95e940fa095c5dbcb52b80f89074fa88e3b11852 |
