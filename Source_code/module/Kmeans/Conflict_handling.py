
def conflict_handling(df):
    handle = df.values.tolist()
    indices_to_drop = set()

    for i in range(len(handle)):
        for j in range(i + 1, len(handle)):
            if handle[i][:-1] == handle[j][:-1]: 
                if handle[i][-1] != handle[j][-1]:
                    indices_to_drop.add(i)
                    indices_to_drop.add(j)

    df = df.drop(index=list(indices_to_drop))
    return df
