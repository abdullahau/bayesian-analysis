fileNames = dir()
Rfiles = grep( "\\.R$" , fileNames , value=TRUE )
for ( fileName in Rfiles ) {
  theText = readLines( fileName )
  grepOut = grep( "read\\.csv" , theText , value=TRUE )
  if ( length( grepOut ) > 0 ) { cat( fileName,"\n" ) }
}
