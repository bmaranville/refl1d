@echo off
echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo dim scriptdir >> CreateShortcut.vbs
echo scriptdir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) >> CreateShortcut.vbs
echo dim dirname >> CreateShortcut.vbs
echo dirname = CreateObject("Scripting.FileSystemObject").GetFileName(scriptdir) >> CreateShortcut.vbs
echo sLinkFile = "%userprofile%\Desktop\" ^& dirname ^&".lnk" >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = scriptdir ^&"\pythonw.exe" >> CreateShortcut.vbs
echo oLink.Arguments = " -m refl1d.main --edit" >> CreateShortcut.vbs
echo oLink.WorkingDirectory = scriptdir >> CreateShortcut.vbs
echo oLink.Description = "Refl1D" >> CreateShortcut.vbs
echo oLink.IconLocation = scriptdir ^&"\refl1d.ico" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs
cscript CreateShortcut.vbs
del CreateShortcut.vbs
