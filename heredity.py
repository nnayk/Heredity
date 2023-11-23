import csv
import itertools
import sys
from operator import mul
from functools import reduce

PROBS = {
    # Unconditional probabilities for having gene
    "gene": {2: 0.01, 1: 0.03, 0: 0.96},
    "trait": {
        # Probability of trait given two copies of gene
        2: {True: 0.65, False: 0.35},
        # Probability of trait given one copy of gene
        1: {True: 0.56, False: 0.44},
        # Probability of trait given no gene
        0: {True: 0.01, False: 0.99},
    },
    # Mutation probability
    "mutation": 0.01,
}


def main():
    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {"gene": {2: 0, 1: 0, 0: 0}, "trait": {True: 0, False: 0}}
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):
        # Check if current set of people violates known information
        fails_evidence = any(
            (
                people[person]["trait"] is not None
                and people[person]["trait"] != (person in have_trait)
            )
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (
                    True
                    if row["trait"] == "1"
                    else False
                    if row["trait"] == "0"
                    else None
                ),
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s)
        for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    def _get_gene_probability(people: dict, person: str, gene: int) -> float:
        """
        Args:
            people: dictionary of people data
            person: specific person
            gene: count of genes (0,1, or 2)

        Returns:
            Probability of the person possessing the given number of genes.
        """

        def _get_pass_probability(gene_count: int) -> float:
            """
            Args:
                gene_count: number of genes (0,1, or 2)

            Returns:
                Probability of passing the gene to the child.
            """
            if gene_count == 0:
                return PROBS["mutation"]
            elif gene_count == 1:
                return 0.5 * (1 - PROBS["mutation"]) + 0.5 * PROBS["mutation"]
            else:
                return 1 - PROBS["mutation"]

        # return top level probability if parents not in dataset
        if (
            people[person]["mother"] is None
            and people[person]["father"] is None
        ):
            return PROBS["gene"][gene]
        # get the gene count for the person's parents
        mother_gene_count = get_gene_count(
            people[person]["mother"], one_gene, two_genes
        )
        father_gene_count = get_gene_count(
            people[person]["father"], one_gene, two_genes
        )
        mother_pass_probability = _get_pass_probability(
            mother_gene_count
        )  # probability of mother passing gene
        father_pass_probability = _get_pass_probability(
            father_gene_count
        )  # probability of father passing gene
        # calculate probability of person having the given number of genes
        if gene == 0:
            return (1 - mother_pass_probability) * (1 - father_pass_probability)
        elif gene == 1:
            return (1 - mother_pass_probability) * father_pass_probability + (
                1 - father_pass_probability
            ) * mother_pass_probability
        else:
            return mother_pass_probability * father_pass_probability

    def _get_trait_probability(people, person, has_trait):
        """
        Given a dictionary of people data, a specific person, and a specific
        gene count, return the probability of the person possessing the given
        number of genes
        """
        pass

    probabilities = []  # contains individual probabilities for each person
    for person in people:
        gene_count, has_trait = 0, False
        gene_probability = 0
        # set the gene value
        gene_count = get_gene_count(person, one_gene, two_genes)
        gene_probability = _get_gene_probability(people, person, gene_count)
        # set the trait value
        if person in have_trait:
            has_trait = True
        trait_probability = _get_trait_probability(people, person, has_trait)
        probabilities.append(gene_probability * trait_probability)
    return reduce(mul, probabilities, 1)


def get_gene_count(person: str, one_gene: set, two_genes: set) -> int:
    """
    Args:
        people: person name
        one_gene: set of people w/1 gene
        two_genes: set of people w/2 genes
    Returns:
        Gene count for the given person
    """
    if person in one_gene:
        return 1
    elif person in two_genes:
        return 2
    else:
        return 0


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    raise NotImplementedError


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
