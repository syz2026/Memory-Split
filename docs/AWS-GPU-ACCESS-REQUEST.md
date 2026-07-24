# AWS GPU v3 temporary operator-role request

Request a time-bounded, federated operator role for
`memorysplit-confirmatory-v3-360m-n10-aws`. The role is for reviewed operations
in `us-east-1` and `us-west-2` only. Require MFA where supported, a short session
duration, cohort session tags, and normal audit logging. Do not issue or store
static access keys.

The EC2 workers use a separate dedicated instance role. The operator role may
pass only that role to EC2; it must not pass itself or any general-purpose role.
Keep package validation and all `msctl` dry runs local. `msctl` does not create
instances, reservations, or Capacity Blocks.

## Requested identity-policy scope

Use the following action groups, narrowed to the two approved Regions and the
named cohort resources wherever the service supports resource-level controls.
The ARN names below are deployment variables, not permission wildcards:
`DedicatedInstanceRoleArn`, `CohortBucketArn`, `CohortObjectArn`,
`CohortKmsKeyArn`, `PrivateEcrRepositoryArn`, and the approved SSM document
ARNs.

### Read-only EC2, quota, and price discovery

- `ec2:DescribeImages`
- `ec2:DescribeInstances`
- `ec2:DescribeInstanceAttribute`
- `ec2:DescribeInstanceTypes`
- `ec2:DescribeInstanceTypeOfferings`
- `ec2:DescribeTags`
- `ec2:DescribeSubnets`
- `ec2:DescribeSecurityGroups`
- `ec2:DescribeVpcs`
- `ec2:DescribeVpcEndpoints`
- `ec2:DescribeCapacityReservations`
- `ec2:DescribeCapacityBlockOfferings`
- `servicequotas:GetServiceQuota`
- `servicequotas:ListServiceQuotas`
- `pricing:DescribeServices`
- `pricing:GetAttributeValues`
- `pricing:GetProducts`

Apply `aws:RequestedRegion` = `us-east-1` or `us-west-2` to regional actions.
AWS Pricing is queried through its supported endpoint and therefore needs a
separate narrowly scoped statement rather than bypassing the EC2 Region guard.
Discovery actions that do not support resource ARNs require `Resource: "*"`;
that does not authorize mutation.

### Explicit-instance EC2 lifecycle

- `ec2:RunInstances`
- `ec2:CreateTags`
- `ec2:DeleteTags`
- `ec2:ModifyInstanceAttribute`
- `ec2:StartInstances`
- `ec2:StopInstances`
- `ec2:TerminateInstances`
- `ec2:DescribeInstances`

Constrain launches to the approved P5/P6 instance types, approved private
subnets and security groups, encrypted volumes, the reviewed immutable AMI ID,
the dedicated instance profile, and cohort request/resource tags. Require
`ec2:CreateAction = RunInstances` on launch-time tagging where applicable.
Start, stop, terminate, instance-attribute, and tag permissions must target
cohort-tagged instances; operators still pass explicit reviewed instance IDs to
every mutating command. `ec2:DeleteTags` is used only by approved `fleet
advance` to remove the prior wave's exact key/value bindings after terminal,
evaluation, and collection evidence has been verified. It is not permission to
retag an active pair manually.

### SSM command and session access

- `ssm:DescribeInstanceInformation`
- `ssm:GetParameter`
- `ssm:GetDocument`
- `ssm:ListDocuments`
- `ssm:CreateDocument`
- `ssm:SendCommand`
- `ssm:GetCommandInvocation`
- `ssm:ListCommands`
- `ssm:ListCommandInvocations`
- `ssm:CancelCommand`
- `ssm:StartSession`
- `ssm:ResumeSession`
- `ssm:TerminateSession`
- `ssmmessages:OpenDataChannel`

`ssm:GetParameter` resolves only the reviewed public DLAMI alias during
discovery. Scope command/session targets to cohort-tagged instances and scope
`ssm:SendCommand` and `ssm:StartSession` to `AWS-RunShellScript`, the fixed
content-addressed MemorySplit command document, and approved session documents.
Permit `ssm:CreateDocument` only for the fixed MemorySplit document name; the
launcher rejects any existing or newly created document whose hash/status
differs. `ssm:ListDocuments`, `ssm:GetDocument`, command-list/get, and
`ssm:CancelCommand` support verification, reconciliation, and explicit
cancellation only. Restrict resume/terminate to sessions owned by the
requesting principal, and scope `ssmmessages:OpenDataChannel` to that
principal's own Session Manager session ARN. No document update/delete and no
public SSH ingress are requested.

### Private ECR push and pull

- `ecr:GetAuthorizationToken`
- `ecr:DescribeRepositories`
- `ecr:DescribeImages`
- `ecr:BatchCheckLayerAvailability`
- `ecr:InitiateLayerUpload`
- `ecr:UploadLayerPart`
- `ecr:CompleteLayerUpload`
- `ecr:PutImage`
- `ecr:BatchGetImage`
- `ecr:GetDownloadUrlForLayer`

Except for `ecr:GetAuthorizationToken`, which does not support repository
resource scope, limit these actions to `PrivateEcrRepositoryArn`. The repository
must already exist, be private and tag-immutable, and use only digest-pinned
runtime references. Repository creation, deletion, policy changes, and tag
mutability changes are not requested.

### Cohort-scoped S3 and KMS

- Bucket: `s3:GetBucketLocation`, `s3:ListBucket`,
  `s3:ListBucketMultipartUploads`
- Cohort prefix objects: `s3:GetObject`, `s3:PutObject`,
  `s3:AbortMultipartUpload`, `s3:ListMultipartUploadParts`
- Cohort key: `kms:DescribeKey`, `kms:Encrypt`, `kms:Decrypt`,
  `kms:GenerateDataKey`

Limit `s3:ListBucket` to the cohort prefixes and all object actions to
`CohortObjectArn`. Require TLS and server-side encryption with
`CohortKmsKeyArn`; constrain KMS use through the approved S3 and ECR services
and encryption context where supported. Bucket deletion, object deletion, ACL
changes, public access, KMS administration, grants, aliases, and key deletion
are not requested.

### Pass only the dedicated EC2 instance role

Request exactly:

- `iam:PassRole` on `DedicatedInstanceRoleArn`
- condition `iam:PassedToService = ec2.amazonaws.com`

Do not request role creation, policy attachment, permission-boundary changes,
or `iam:PassRole` on wildcard resources. The dedicated instance role should
independently receive only private ECR pull, cohort-prefix S3/KMS use, and the
managed-instance permissions needed for SSM.

## Separate P6 Capacity Block purchase elevation

The baseline operator role should include
`ec2:DescribeCapacityBlockOfferings` but should not include
`ec2:PurchaseCapacityBlock`.

A P6 Capacity Block is paid and non-cancellable. For a purchase, require a
separate ticket or approval artifact that records the exact offering ID,
instance type, instance count, Region, start/end, duration, currency, and exact
total price. A different offering or price invalidates approval. After a
read-only discovery receipt is reviewed:

1. Temporarily add or session-scope `ec2:PurchaseCapacityBlock` in the approved
   Region. The API does not provide useful resource-level scoping, so use
   `Resource: "*"` only in this isolated statement.
2. Run the exact purchase command with its dry-run control and review the
   expected permission result.
3. Reconfirm the exact-price approval, then perform one explicit purchase.
4. Remove the purchase elevation immediately and retain the approval, dry-run,
   and purchase receipts.

Package approval, provider selection, `RunInstances` permission, or an `msctl`
approval must never be treated as Capacity Block purchase approval.

## Known `InternSandboxBoundary` limitation

The known `InternSandboxBoundary` permission boundary can deny actions even
when this temporary role's identity policy allows them. An identity policy
cannot override that boundary. Capacity Block purchase, tightly scoped
`iam:PassRole`, KMS use, ECR upload, or other requested actions may therefore
remain denied until the platform owner updates the boundary or supplies a
separately governed role outside that boundary.

Do not broaden this request to work around the boundary. Have the IAM/platform
owner compare the requested actions with the effective boundary, approve only
the missing least-privilege actions, and test first with read-only calls and AWS
dry-run controls. No account identifier or personal contact is needed in this
template.

## Expiry and removal

Set an explicit expiry covering qualification, the reviewed run window, and
teardown only. At expiry, remove the temporary operator policy and any Capacity
Block elevation, close active SSM sessions, and retain CloudTrail plus cohort
approval/operation receipts according to the study retention policy. Do not
delete durable cohort artifacts, ECR digests, or KMS keys as part of access
revocation.
