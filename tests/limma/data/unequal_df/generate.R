# Run from the repository root with R + limma + jsonlite installed.
# No Python implementation is used to generate these expected values.
# Optional first argument: directory with the canonical R source files below.
library(limma)
library(jsonlite)
revision <- '02a9dbd9c97d0ac9c5003c171cb1a3809cd1f16e'
files <- c('fitFDistUnequalDF1.R','fitFDist.R','squeezeVar.R','ebayes.R',
           'loessFit.R','weightedLowess.R','chooseLowessSpan.R')
args <- commandArgs(trailingOnly=TRUE)
source.dir <- if (length(args)) args[1] else tempfile('limma-source-')
if (!dir.exists(source.dir)) dir.create(source.dir)
reference <- new.env(parent=asNamespace('limma'))
for (file in files) {
    path <- file.path(source.dir,file)
    if (!file.exists(path)) download.file(paste0('https://raw.githubusercontent.com/bioc/limma/',revision,'/R/',file),path,quiet=TRUE)
    sys.source(path,envir=reference)
}
root <- 'tests/limma/data/unequal_df'
x <- c(.12,.2,.4,.8,1.5,3.2,.35,2.4,5,.7,8,1.1)
df <- c(1,2,3,4,5,6,8,10,12,2,4,9)
cases <- list(
    equal=list(x=x,df=rep(6,12)),
    unequal=list(x=x,df=df),
    equal_trend=list(x=x,df=rep(6,12),covariate=seq_len(12)),
    unequal_trend=list(x=rep(x,2)*exp(rep(1:6,each=4)/5),df=rep(df,2),covariate=rep(1:6,each=4)),
    fractional=list(x=x,df=c(.001,.009,.01,.05,.1,.5,1,2.5,8.2,20,100,1e4)),
    invalid=list(x=c(.1,.5,2,5,1,NA,0,-2,-Inf,3),df=c(2,3,4,8,12,3,5,6,7,0)),
    boundary=list(x=rep(1,12),df=df),
    zero_df=list(x=c(.1,.5,2,8,NA),df=c(4,4,4,4,0)),
    weighted=list(x=x,df=df,prior_weights=c(0,1,.5,2,1,1,.2,1,3,1,1,1))
)
for (name in names(cases)) {
    item <- cases[[name]]
    item$fit <- reference$fitFDistUnequalDF1(item$x,item$df,covariate=item$covariate,prior.weights=item$prior_weights)
    if (is.null(item$prior_weights)) {
        item$default <- reference$squeezeVar(item$x,item$df,covariate=item$covariate)
        item$legacy <- reference$squeezeVar(item$x,item$df,covariate=item$covariate,legacy=TRUE)
        item$modern <- reference$squeezeVar(item$x,item$df,covariate=item$covariate,legacy=FALSE)
        if (!is.null(item$covariate)) item$span <- reference$squeezeVar(item$x,item$df,covariate=item$covariate,span=.6,legacy=TRUE)
    }
    cases[[name]] <- item
}
# A single deterministic trend with >200 observations exercises seed interpolation.
i <- seq_len(211)
smoother <- list(x=sort(c(seq(0,1,length.out=70),seq(3,5,length.out=70),seq(10,11,length.out=71))),
                 y=sin(i/7)+i/300,weights=rep(c(.01,1,2,8),length.out=211),span=.6)
smoother$fitted <- reference$loessFit(smoother$y,smoother$x,weights=smoother$weights,min.weight=1e-8,max.weight=100,span=.6,iterations=1)$fitted
small_trends <- list(
    tied=list(x=c(0,0,0,1,1,1),y=c(1,2,4,1,3,7),weights=rep(1,6)),
    constant=list(x=rep(1,6),y=c(1,2,4,1,3,7),weights=rep(1,6)),
    linear=list(x=c(0,1,2),y=c(1,3,2),weights=c(.5,1,2)))
for (name in names(small_trends)) {
    z <- small_trends[[name]]
    z$fitted <- reference$loessFit(z$y,z$x,weights=z$weights,span=.6,iterations=1)$fitted
    small_trends[[name]] <- z
}
edge_cases <- list(
    no_information=list(x=c(0,0,1),df=c(2,3,0)),
    two_informative=list(x=c(1,2,NA),df=c(2,3,4)),
    infinite_variance=list(x=c(1,2,3,Inf),df=c(2,3,4,5)))
for (name in names(edge_cases)) {
    z <- edge_cases[[name]]
    z$fit <- suppressWarnings(reference$fitFDistUnequalDF1(z$x,z$df))
    edge_cases[[name]] <- z
}
read <- function(name) as.matrix(read.csv(file.path(root,paste0('missing_',name,'.csv')),row.names=1,check.names=FALSE))
y <- read('expression')
design <- read('design')
contrasts <- read('contrasts')
fit <- contrasts.fit(lmFit(y,design),contrasts)
missing <- list(coefficients=fit$coefficients,stdev_unscaled=fit$stdev.unscaled,
                sigma=fit$sigma,df=fit$df.residual,Amean=fit$Amean,
                genes=rownames(fit),contrasts=colnames(fit))
for (name in c('default','legacy','trend','span')) {
    out <- reference$eBayes(fit,legacy=if(name %in% c('legacy','span')) TRUE else NULL,
                            trend=(name %in% c('trend','span')),span=if(name=='span') .6 else NULL)
    missing[[name]] <- list(df_prior=out$df.prior,s2_prior=out$s2.prior,s2_post=out$s2.post,
                           df_total=out$df.total,t=out$t,p_value=out$p.value)
}
metadata <- list(R=R.version.string,limma=as.character(packageVersion('limma')),source_revision=revision,
                 source_md5=as.list(setNames(tools::md5sum(file.path(source.dir,files)),files)))
write_json(list(metadata=metadata,cases=cases,smoother=smoother,small_trends=small_trends,edge_cases=edge_cases,missing=missing),
           file.path(root,'reference.json'),digits=NA,pretty=TRUE,na='string',auto_unbox=TRUE)
