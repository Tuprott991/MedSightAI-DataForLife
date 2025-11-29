import { KeyFindings } from './KeyFindings';
import { SuspectedDisease } from './SuspectedDisease';

export const AnalysisTab = ({ findings, suspectedDiseases, onFindingClick }) => {
    return (
        <div className="space-y-4">
            <SuspectedDisease diseases={suspectedDiseases || []} />
            <KeyFindings findings={findings} onFindingClick={onFindingClick} />
        </div>
    );
};
