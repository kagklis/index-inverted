@echo off

set TAGDIR=C:\TreeTagger

set BIN=%TAGDIR%\bin
set CMD=%TAGDIR%\cmd
set LIB=%TAGDIR%\lib
set TAGOPT=%LIB%\spanish.par -token -lemma -sgml -no-unknown

if "%2"=="" goto label1
perl %CMD%\tokenize.pl -i -a %LIB%\spanish-abbreviations %1 | %BIN%\tree-tagger %TAGOPT% > %2
goto end

:label1
if "%1"=="" goto label2
perl %CMD%\tokenize.pl -i -a %LIB%\spanish-abbreviations %1 > t1
perl %CMD%\mwl-lookup.perl  -f %LIB%\spanish-mwls t1 > t2
%BIN%\tree-tagger %TAGOPT% t2
del t1
del t2
goto end

:label2
echo.
echo Usage: tag-spanish file {file}
echo.

:end
