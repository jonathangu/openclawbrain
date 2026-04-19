# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-15f74481afa0ad3c49942a752d93fa21610759dcd0f5184c05ee667b747607b5`
- fixture hash: `sha256-27569dfe07b6cf66e357fc072347afe0c073b0dd225ff6f7f6dbd4f6b53bd5c5`
- score hash: `sha256-130a17fe22352e802ed65249b3fb8487be1715d4c525b07a0e301a85f88fcdcc`
- bundle hash: `sha256-ff02daeaa3b08692b92f3f5a2855275e43eab05469e5591691d8edde3061db29`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d9780bcb02b4dddac9cfba41582ad72477a9d4e9b030a1ad3ced919c347c5d08 |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-a355516b8e5b8aa496256577ec80f625725547c6a5d7a8445ebf53234ef9bbc9 |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-b4dc03d6f6b5f48deecae43d5fab0c6e7f7ababa224851c030eb81e1710c24da |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-8983eeec283d7a5983e49d98fafc3a480695cf5b9c5a221ac085f1a05bc522a9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-c4fc9b91 | sha256-96bb82bedeaf08585e7fd77184a51b5b22c34c06a4a73b95b00baae9839b7b31 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-c4fc9b91 | sha256-b7d45001e4850935b6ed13c72d78bdd8fee75776472545eb8a272b556960f0f6 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-c4fc9b91 | sha256-96bb82bedeaf08585e7fd77184a51b5b22c34c06a4a73b95b00baae9839b7b31 |
