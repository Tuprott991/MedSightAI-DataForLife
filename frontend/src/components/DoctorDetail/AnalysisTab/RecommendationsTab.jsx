import { Recommendations } from './Recommendations';

export const RecommendationsTab = ({ recommendations }) => {
    return (
        <div>
            <Recommendations recommendations={recommendations} />
        </div>
    );
};
