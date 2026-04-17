# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-044`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0ad01cba10197b8b05f27f973f11a9d5ae36ed08ae5a28bf29003ec749435fdb`
- fixture hash: `sha256-6830cd222cfba386b49de3a4d46620d84d8c333ee746eafd9c6c6f8ee2dfa95c`
- score hash: `sha256-4a51e05729a1a67669a43edd1bd3af8baac4f9030cf22d43da1b77d287312d7b`
- bundle hash: `sha256-4a38ac4ac9b477dbcf1d1e12568bd3d5880231a323200ac859562ddc644ad4f4`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-490c3e2fb439b8787ca9fe3cb573a9c587cc3abe414e2faf637ea2b84d91d268 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2bbadcaf0202e256e84c86fbc2fa8fe6a43cc8312b1b2ade5f544f3ab1b6e845 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a56686ca507254a958d9b304ba1c0e5a27764adaac7a73ab23c78fbcfae055dc |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-e7d5c20cf9929832f4f7e36393b0f4f51e779fc1d115d807084b92bbf51e3d95 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3dc31c0c | sha256-9b07bfa2e0c1a73d040f90e9aae42c1d967e745df6c8c98fbb8b349f57e7c503 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3dc31c0c | sha256-8407452bc98c25e00f981aaf29bbabae1ccb8b55203b61f3d072e2e4ab15b871 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-5e53bfd5 | sha256-df5c64520325ebf4d788fbeeef357cf8174d36379087a8e12e4ab3fade75517d |
