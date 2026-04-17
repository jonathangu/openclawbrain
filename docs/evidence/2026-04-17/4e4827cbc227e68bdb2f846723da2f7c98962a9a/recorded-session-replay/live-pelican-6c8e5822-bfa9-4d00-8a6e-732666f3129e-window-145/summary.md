# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-145`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8588f62e6cb39b6bbebdb00e938513a4cbaa506b41be87532a11b4304976dd66`
- fixture hash: `sha256-523f979d52f465f7796de01a235f1b7bbea1b624b0a2f4aa71ab4b02e1ae0958`
- score hash: `sha256-f3d27805fdac20c68a275606db7540c7cd624159de43c98522ebafad3898eeaf`
- bundle hash: `sha256-8f092f26fdc788b1ec602115ef870e45850e002b4511f34626d076adb1ec2a6a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-172463d69f5ae184c08f379b77a680b592819857917ad8f3596af66f22037f0d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e3052d707f1388174f34455d153e62cba8ac080eaa26d7f9d5730c9d60372f20 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3b13bd5fd9bc64fc8f25a96343e3702a1f1e7ec9746a1787e9e09291a93f6c3d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-ed4f80afc6428cecf63dc50041bf80b8406ceedbab3c8fd21c02f50754bc2ec8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-586ebb85 | sha256-ada23daa25022dfafd5fe90315542a2b95368826449f73e31560e9d6ced8ffae |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-586ebb85 | sha256-519d91bf1be6d5c78240319a7d917f737a40c47a8518128944f1c99cd1f0fb6f |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-78a39814 | sha256-9657183f70d537e0f36a6682ecfd25095f84651357a9ecea0805aa0efc0cdf57 |
