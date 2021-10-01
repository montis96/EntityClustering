{ echo "Secondo blocco";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.018" --randomly --seed="42" ;
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.013" --randomly --seed="42" ;
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" ; } &
sleep 3
{ echo "Secondo blocco";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" --entropy="26";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" --entropy="24";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" --entropy="22"; } &
sleep 3
{ echo "Secondo blocco";
  python ./evolving_clustering/script_ubuntu.py --step=30 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" ;
  python ./evolving_clustering/script_ubuntu.py --step=70 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" ;
  python ./evolving_clustering/script_ubuntu.py --step=120 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" ; } &
