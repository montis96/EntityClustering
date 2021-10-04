{ echo "Secondo blocco";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.014" --randomly --seed="42" ;
  python ./evolving_clustering/script_ubuntu.py --step=1 --first_threshold="0.03678974213038678" --second_threshold="0.018" --randomly --seed="42" ;} &
sleep 3
{ echo "Secondo blocco";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" --entropy="26";
  python ./evolving_clustering/script_ubuntu.py --step=10 --first_threshold="0.03678974213038678" --second_threshold="0.14" --randomly --seed="42" --entropy="25"; } &
sleep 3
{ echo "Secondo blocco";
  python ./evolving_clustering/script_ubuntu.py --step=30 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" ;
  python ./evolving_clustering/script_ubuntu.py --step=70 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" ;
  python ./evolving_clustering/script_ubuntu.py --step=120 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" ; } &
sleep 3
{ echo "Secondo blocco";
  python ./evolving_clustering/script_ubuntu.py --step=30 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" --entropy="100000"; ;
  python ./evolving_clustering/script_ubuntu.py --step=70 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" --entropy="12"; ;
  python ./evolving_clustering/script_ubuntu.py --step=120 --first_threshold="0.03678974213038678" --second_threshold="0.0155" --randomly --seed="42" ; --entropy="30"; } &