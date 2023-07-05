#!/usr/bin/env python3

# Constants needed for ablation calculations
# From Bionumbers
CELL_DRY_MASS = 15e-12  # g
# Computed from isa reactions in Yeast8 model,
# see molecular weights notebook
MW_CARB = 350.37049806068  # g/mol
MW_DNA = 3.898762476
MW_RNA = 64.04235752722991
MW_PROTEIN = 504.37442340123584
MW_COFACTOR = 4.8326512432304005
MW_ION = 2.4815607543700002
MW_LIPID = 31.5659867112958
MW_BIOMASS = 961.5662401740419

# Defaut IDs for growth and biomass reactions for batch Yeast8 model
GROWTH_ID = "r_2111"
BIOMASS_ID = "r_4041"
# IDs of pseudoreactions connected to biomass.
# Doesn't contain biomass itself (though it is in the 'Growth' subsystem)
# to make coding easier.
GROWTH_SUBSYSTEM_IDS = [
    "r_2108",
    "r_4047",
    "r_4048",
    "r_4050",
    "r_4049",
    "r_4598",
    "r_4599",
]

# List genes to delete and exchange reactions to add for each auxotroph
AuxotrophProperties = namedtuple(
    "AuxotrophProperties", ["genes_to_delete", "exch_to_add"]
)

AUXOTROPH_DICT = {
    "BY4741": AuxotrophProperties(
        ["YOR202W", "YCL018W", "YLR303W", "YEL021W"],
        [
            "r_1893",
            "r_1899",
            "r_1902",
            "r_2090",
            "r_1893_REV",
            "r_1899_REV",
            "r_1902_REV",
            "r_2090_REV",
        ],
    ),
    "BY4742": AuxotrophProperties(
        ["YOR202W", "YCL018W", "YBR115C", "YEL021W"],
        [
            "r_1893",
            "r_1899",
            "r_1900",
            "r_2090",
            "r_1893_REV",
            "r_1899_REV",
            "r_1900_REV",
            "r_2090_REV",
        ],
    ),
    "BY4741_orig": AuxotrophProperties(
        ["YOR202W", "YCL018W", "YLR303W", "YEL021W"],
        [
            "EX_his__L_e",
            "EX_met__L_e",
            "EX_leu__L_e",
            "EX_ura_e",
        ],
    ),
    "BY4742_orig": AuxotrophProperties(
        ["YOR202W", "YCL018W", "YBR115C", "YEL021W"],
        [
            "EX_his__L_e",
            "EX_met__L_e",
            "EX_lys__L_e",
            "EX_ura_e",
        ],
    ),
}

# TODO: Add lookup tables:
# - Carbon sources/other nutrients
# - Genes (systematic vs common)
