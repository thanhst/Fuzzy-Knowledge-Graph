function rules_index_reduced = reduce_rules(train_data_muy, train_data_alpha)

test_num = size(train_data_muy,1);
threshold = 0.2;
index = 2; %Initiate value to pass the condition of WHILE Loop
train_data_muy_updated = train_data_muy;
train_data_alpha_updated = train_data_alpha;
for i=1:test_num
    train_data_index_updated(i) = i;
end
cluster_index = 1;
candidates_muy = [];
candidates_alpha = [];
member_count = 1;
rules_index = 1;
rules_index_reduced = [];
while index > 1 && member_count > 0
    index = 1;
    member_count = 0;
    D_Sim_temp = calculate_D_Similarity(train_data_muy_updated, train_data_alpha_updated, train_data_muy_updated(1,:), train_data_alpha_updated(1,:))
    test_num_update = size(train_data_muy_updated,1);
    attribute_num_update = size(train_data_muy_updated,2);
    train_data_muy_updated_temp = [];
    train_data_alpha_updated_temp = [];
    rules_index_updated= [];
    if test_num_update > 1
        for i=1:test_num_update
            if D_Sim_temp(i) < threshold
                for j=1:attribute_num_update
                    train_data_muy_updated_temp(index,j) = train_data_muy_updated(i,j);
                    train_data_alpha_updated_temp(index,j) = train_data_alpha_updated(i,j);
                end
                rules_index_updated(index) = train_data_index_updated(i);
                index = index + 1;
            else
                if member_count < 1
                    member_count = member_count + 1;
                    for j=1:attribute_num_update
                        candidates_muy(cluster_index,j) = train_data_muy_updated(i,j);
                        candidates_alpha(cluster_index,j) = train_data_alpha_updated(i,j);
                    end
                    rules_index = train_data_index_updated(i);
                    rules_index_reduced = [rules_index_reduced rules_index];
                    cluster_index = cluster_index + 1;
                end
            end
        end
        if member_count < 1
            for j=1:attribute_num_update
                candidates_muy(cluster_index,j) = train_data_muy_updated_temp(1,j);
                candidates_alpha(cluster_index,j) = train_data_alpha_updated_temp(1,j);
            end
            train_data_muy_updated_temp(1,:) = [];
            train_data_alpha_updated_temp(1,:) = [];
            rules_index = rules_index_updated(1);
            rules_index_reduced = [rules_index_reduced rules_index];
            cluster_index = cluster_index + 1;
            rules_index_updated(1) = [];
            member_count = member_count + 1;
        end
    end
    if size(train_data_muy_updated_temp,1) > 1
        train_data_muy_updated = train_data_muy_updated_temp;
        train_data_alpha_updated = train_data_alpha_updated_temp;
        for i=1:(index-2)
            train_data_index_updated(i) = rules_index_updated(i);
        end
    else
        for j=1:attribute_num_update
            candidates_muy(cluster_index,j) = train_data_muy_updated_temp(1,j);
            candidates_alpha(cluster_index,j) = train_data_alpha_updated_temp(1,j);
        end
        rules_index = rules_index_updated(1);
        rules_index_reduced = [rules_index_reduced rules_index];
        member_count = 0;
    end
end

% if size(candidates_muy) == 0
    % fprintf('  candidates_muy is empty...\n');
% else
    % candidates_muy
% end
% if size(candidates_alpha) == 0
    % fprintf('  candidates_alpha is empty...\n');
% else
    % candidates_alpha
% end