cd ../
if [ -d NASFolder ] 
then
mkdir NASFolder
fi
sudo sshfs osvallo@cvblab.synology.me:/CVBLab/Proyectos/DIPSY/INVESTIGACION/Speech/data_folder /home/oscar/dipsy/ssast/NASFolder -o allow_other -p 1098