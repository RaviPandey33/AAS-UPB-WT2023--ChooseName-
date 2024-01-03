library(smoof)
library(flacco)

listAvailableFeatureSets()
feature_set <- "ela_meta"

fn <- makeBBOBFunction(dimensions = 2L, fid = 2L, iid = 1L)
fn(c(1, 1))

control <- list("init_sample.lower" = -5, "init_sample.upper" = 5)
X <- createInitialSample(n.obs = 500, dim = 2, control = control)
y <- apply(X, 1, fn)

feat.object <- createFeatureObject(X = X, y = y)

featureSet <- calculateFeatureSet(feat.object, set = feature_set)
featureSet
