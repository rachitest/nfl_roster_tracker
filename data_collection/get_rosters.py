import pandas as pd
import sqlalchemy as sql

class Team:
    def __init__(self, name, abbr, site):
        self.name = name
        self.abbr = abbr
        self.site = site
    
    def get_roster(self):
        url = self.site + "team/players-roster"
        df = pd.read_html(url)
        
        roster = pd.concat(df)
        roster = roster.reset_index(drop=True)

        return roster


def roster_dict(teams):
    league_roster = {}
    for team in teams:
        league_roster[f'{team.abbr}'] = team.get_roster()
    
    return league_roster


def excel_out(league_roster):
    writer = pd.ExcelWriter('NFL_rosters.xlsx', engine='xlsxwriter')

    for key, value in league_roster.items():
        value.to_excel(writer, sheet_name=key)

    writer.save()


def sqlite_out(league_roster):
    engine = sql.create_engine('sqlite:///NFL_rosters.db', echo=False)

    for key, value in league_roster.items():
        value.to_sql(key, con=engine, if_exists='replace')


if __name__ == '__main__':

    ARI = Team("Arizona Cardinals", "ARI", "https://www.azcardinals.com/")
    ATL = Team("Atlanta Falcons", "ATL", "https://www.atlantafalcons.com/")
    BAL = Team("Baltimore Ravens", "BAL", "https://www.baltimoreravens.com/")
    BUF = Team("Buffalo Bills", "BUF", "https://www.buffalobills.com/")
    CAR = Team("Carolina Panthers", "CAR" , "https://www.panthers.com/")
    CHI = Team("Chicago Bears", "CHI", "https://www.chicagobears.com/")
    CIN = Team("Cincinnati Bengals", "CIN", "https://www.bengals.com/")
    CLE = Team("Cleveland Browns", "CLE", "https://www.clevelandbrowns.com/")
    DAL = Team("Dallas Cowboys", "DAL", "https://www.dallascowboys.com/")
    DEN = Team("Denver Broncos", "DEN", "https://www.denverbroncos.com/")
    DET = Team("Detroit Lions", "DET", "https://www.detroitlions.com/")
    GB = Team("Green Bay Packers", "GB", "https://www.packers.com/")
    HOU = Team("Houston Texans", "HOU", "https://www.houstontexans.com/")
    IND = Team("Indianapolis Colts", "IND", "https://www.colts.com/")
    JAX = Team("Jacksonville Jaguars", "JAX", "https://www.jaguars.com/")
    KC = Team("Kansas City Chiefs", "KC", "https://www.chiefs.com/")
    LAC = Team("Los Angeles Chargers", "LAC", "https://www.chargers.com/")
    LAR = Team("Los Angeles Rams", "LAR", "https://www.therams.com/")
    LV = Team("Las Vegas Raiders", "LV", "https://www.raiders.com/")
    MIA = Team("Miami Dolphins", "MIA", "https://www.miamidolphins.com/")
    MIN = Team("Minnesota Vikings", "MIN", "https://www.vikings.com/")
    NE = Team("New England Patriots", "NE", "https://www.patriots.com/")
    NO = Team("New Orleans Saints", "NO", "https://www.neworleanssaints.com/")
    NYG = Team("New York Giants", "NYG", "https://www.giants.com/")
    NYJ = Team("New York Jets", "NYJ", "https://www.newyorkjets.com/")
    PHI = Team("Philadelphia Eagles", "PHI", "https://www.philadelphiaeagles.com/")
    PIT = Team("Pittsburgh Steelers", "PIT", "https://www.steelers.com/")
    SEA = Team("Seattle Seahawks", "SEA", "https://www.seahawks.com/")
    SF = Team("San Francisco 49ers", "SF", "https://www.49ers.com/")
    TB = Team("Tampa Bay Buccaneers", "TB", "https://www.buccaneers.com/")
    TEN = Team("Tennessee Titans", "TEN", "https://www.titansonline.com/")
    WAS = Team ("Washington Football Team", "WAS", "https://www.redskins.com/")
    
    teams = [ARI, ATL, BAL, BUF, CAR, CHI, CIN, CLE, DAL, DEN, DET, GB, HOU, IND, JAX, KC, LAC, LAR, LV, MIA, MIN, NE, NO, NYG, NYJ, PHI, PIT, SEA, SF, TB, TEN, WAS]

    league_roster = roster_dict(teams)
    excel_out(league_roster)
    sqlite_out(league_roster)