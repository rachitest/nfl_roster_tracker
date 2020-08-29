Set-Location "G:\My Drive\Backup\Documents\Python Projects\nfl_roster_tracker\data_collection"

poetry shell

python get_rosters.py

git add NFL_rosters.db; git commit -m "Update NFL_rosters.db"; git push origin origin nfl_roster_tracker-master